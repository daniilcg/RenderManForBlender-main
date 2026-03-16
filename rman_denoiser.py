import os
import numpy as np
from collections import OrderedDict

from rfb_utils.envconfig_utils import envconfig
from rfb_utils import scene_utils


class RmanDenoiser:

    def __init__(self, stats_mgr):

        self.stats_mgr = stats_mgr
        self.denoiser = None

        self.width = -1
        self.height = -1
        self.asymmetry = 0.0
        self.use_color_pass = False

        self.topology = None

        # reusable buffers
        self.features = {}
        self.asymmetry_buffer = None
        self.divide_buffer = None

    def bootstrap(self, width, height, asymmetry, use_color_pass):

        try:
            import QuicklyNoiseless as qn
        except ImportError:
            return

        self.width = width
        self.height = height
        self.asymmetry = asymmetry
        self.use_color_pass = use_color_pass

        rman = envconfig().rmantree

        if asymmetry > 0:
            param = os.path.join(rman, "lib", "denoise", "14433-renderman.param")
            topo = os.path.join(rman, "lib", "denoise", "full_w1_5s_asym.topo")
        else:
            param = os.path.join(rman, "lib", "denoise", "20973-renderman.param")
            topo = os.path.join(rman, "lib", "denoise", "full_w1_5s_sym_gen2.topo")

        self.topology = topo

        self.denoiser = qn.Denoiser(height, width, param, topo)
        self.denoiser.enableWarping(-1.0, 1.0, False)

        # preallocated buffers
        self.asymmetry_buffer = np.full(
            (height, width, 1),
            asymmetry,
            dtype=np.float32
        )

        self.divide_buffer = np.zeros(
            (height, width, 1),
            dtype=np.float32
        )

    def _prepare_features(self, base, data, variance):

        f = self.features
        f.clear()

        f["albedo"] = base["albedo"]
        f["albedoVariance"] = base["albedoVariance"]
        f["normal"] = base["normal"]
        f["normalVariance"] = base["normalVariance"]
        f["sampleCount"] = base["sampleCount"]

        f["data"] = data
        f["dataVariance"] = variance

        if "asym" in self.topology and data is not None:
            f["asymmetry"] = self.asymmetry_buffer

        if "full" in self.topology and "gen2" not in self.topology:
            f["divideAlbedo"] = self.divide_buffer

        return f

    def _compute_weights(self, features):

        self.denoiser.setFeatures(features, 0)
        self.denoiser.computeWeights()

    def _apply(self, data):

        return self.denoiser.applyWeights([data])

    def denoise(self, passes, render, render_border):

        if not self.denoiser:
            return None

        variance = passes.get("variance", {})

        passInput = variance.get("input")
        passInputVar = variance.get("input_variance")

        passAlbedo = variance.get("albedo")
        passAlbedoVar = variance.get("albedo_variance")

        passNormal = variance.get("normal")
        passNormalVar = variance.get("normal_variance")

        passSample = variance.get("sample_count")

        passDiffuse = variance.get("diffuse")
        passDiffuseVar = variance.get("diffuse_variance")

        passSpecular = variance.get("specular")
        passSpecularVar = variance.get("specular_variance")

        passAlpha = variance.get("alpha")
        passAlphaVar = variance.get("alpha_variance")

        base_features = {
            "albedo": passAlbedo,
            "albedoVariance": passAlbedoVar,
            "normal": passNormal,
            "normalVariance": passNormalVar,
            "sampleCount": passSample
        }

        denoised_passes = OrderedDict()

        self.stats_mgr.draw_message("Denoising (beauty)")

        # --- beauty pass ---

        if self.use_color_pass:

            f = self._prepare_features(base_features, passInput, passInputVar)

            self._compute_weights(f)

            beauty = self._apply(passInput)

        else:

            f = self._prepare_features(base_features, passDiffuse, passDiffuseVar)

            self._compute_weights(f)

            diffuse = self._apply(passDiffuse)
            specular = self._apply(passSpecular)

            beauty = diffuse + specular

        # --- alpha ---

        f = self._prepare_features(base_features, passAlpha, passAlphaVar)

        self._compute_weights(f)

        alpha = self._apply(passAlpha)

        # --- borders ---

        use_border = render.use_border and not render.use_crop_to_border

        if render_border:
            sy, ey, sx, ex = render_border
        else:
            _, _, sx, ex, sy, ey = scene_utils.get_render_borders(
                render,
                self.height,
                self.width
            )

        if use_border:
            beauty = beauty[sy:ey, sx:ex, :]
            alpha = alpha[sy:ey, sx:ex, :]

        pixels = (ey - sy) * (ex - sx)

        beauty = beauty.reshape(pixels, 3)
        alpha = alpha[..., 0:1].reshape(pixels, 1)

        denoised_passes["beauty"] = np.concatenate((beauty, alpha), axis=1)

        # --- other passes ---

        for name, p in passes.items():

            if name == "variance":
                continue

            if p["num_channels"] == 3:

                f = self._prepare_features(base_features, passInput, passInputVar)

            else:

                f = self._prepare_features(base_features, passAlpha, passAlphaVar)

            self._compute_weights(f)

            result = self._apply(p["input"])

            if use_border:
                result = result[sy:ey, sx:ex, :]

            result = result.reshape(pixels, p["num_channels"])

            denoised_passes[name] = result

        self.stats_mgr._progress = 100

        return denoised_passes