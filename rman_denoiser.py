import os
from .rfb_utils.envconfig_utils import envconfig
from .rfb_utils import scene_utils
from collections import OrderedDict
import numpy as np

class RmanDenoiser:

    def __init__(self, stats_mgr):
        self.stats_mgr = stats_mgr
        self.width = -1
        self.height = -1
        self.asymmetry = 0.0
        self.use_color_pass = False

        # Denoiser
        self.denoiser = None

    def bootstrap(self, width, height, asymmetry, use_color_pass):
        qn = None
        try:
            import QuicklyNoiseless as qn
        except ImportError:
            self.denoiser = None
            return 

        self.width = width
        self.height = height
        self.asymmetry = asymmetry
        self.use_color_pass = use_color_pass
        if asymmetry > 0.0:
            self.parameters = os.path.join(envconfig().rmantree, "lib", "denoise", "14433-renderman.param")
            self.topology = os.path.join(envconfig().rmantree, "lib", "denoise", "full_w1_5s_asym.topo")
        else:
            self.parameters = os.path.join(envconfig().rmantree, "lib", "denoise", "20973-renderman.param")
            self.topology = os.path.join(envconfig().rmantree, "lib", "denoise", "full_w1_5s_sym_gen2.topo")

        self.denoiser = qn.Denoiser(height, width, self.parameters, self.topology)
        self.denoiser.enableWarping(-1.0, 1.0, False)           

    def denoise(self, passes, render, render_border):
        if self.denoiser is None:
            return None
        
        total = len(passes)
        finished = 0
        self.stats_mgr._progress = int(0)
        self.stats_mgr.draw_message("Denoising (beauty)")

        denoised_passes = OrderedDict()

        # Extract the variance channels
        passInput = passes.get("variance").get("input", None)
        passNormal = passes.get("variance").get("normal", None)
        passNormalVariance = passes.get("variance").get("normal_variance", None)
        passInputVariance = passes.get("variance").get("input_variance", None)
        passAlbedo = passes.get("variance").get("albedo", None)
        passAlbedoVariance = passes.get("variance").get("albedo_variance", None)
        passSampleCount = passes.get("variance").get("sample_count", None)
        passAlpha = passes.get("variance").get("alpha", None)
        passAlphaVariance = passes.get("variance").get("alpha_variance", None)
        passDiffuse = passes.get("variance").get("diffuse", None)
        passDiffuseVariance = passes.get("variance").get("diffuse_variance", None)
        passSpecular = passes.get("variance").get("specular", None)
        passSpecularVariance = passes.get("variance").get("specular_variance", None)


        # Build features for beauty pass
        features = {
            'albedo': passAlbedo,
            'albedoVariance': passAlbedoVariance,
            'normal': passNormal,
            'normalVariance': passNormalVariance,
            'sampleCount': passSampleCount
        }
        if self.use_color_pass:
            features["data"] = passInput
            features["dataVariance"] = passInputVariance
            if "asym" in self.topology and features["data"] is not None:
                asymmetry = np.broadcast_to(self.asymmetry, shape=features["data"].shape[:-1] + (1,))
                features["asymmetry"] = asymmetry
            if "full" in self.topology and "gen2" not in self.topology and features["data"] is not None:
                divideAlbedo = np.broadcast_to(0.0, shape=features["data"].shape[:-1] + (1,))
                features["divideAlbedo"] = divideAlbedo
            self.denoiser.setFeatures(features, 0)
            self.denoiser.computeWeights()
            denoisedBeauty = self.denoiser.applyWeights([passInput])
        else:
            features["data"] = passDiffuse
            features["dataVariance"] = passDiffuseVariance
            if "asym" in self.topology and features["data"] is not None:
                asymmetry = np.broadcast_to(self.asymmetry, shape=features["data"].shape[:-1] + (1,))
                features["asymmetry"] = asymmetry
            if "full" in self.topology and "gen2" not in self.topology and features["data"] is not None:
                divideAlbedo = np.broadcast_to(0.0, shape=features["data"].shape[:-1] + (1,))
                features["divideAlbedo"] = divideAlbedo
            self.denoiser.setFeatures(features, 0)
            self.denoiser.computeWeights()
            denoisedDiffuse = self.denoiser.applyWeights([passDiffuse])

            features["data"] = passSpecular
            features["dataVariance"] = passSpecularVariance
            self.denoiser.setFeatures(features, 0)
            self.denoiser.computeWeights()
            denoisedSpecular = self.denoiser.applyWeights([passSpecular])

            denoisedBeauty = denoisedDiffuse + denoisedSpecular

        # Alpha pass (new features dict)
        features_alpha = {
            'albedo': passAlbedo,
            'albedoVariance': passAlbedoVariance,
            'normal': passNormal,
            'normalVariance': passNormalVariance,
            'sampleCount': passSampleCount,
            'data': passAlpha,
            'dataVariance': passAlphaVariance
        }
        self.denoiser.setFeatures(features_alpha, 0)
        self.denoiser.computeWeights()
        denoisedAlpha = self.denoiser.applyWeights([passAlpha])
        

        # check for crop windows and borders
        use_border = render.use_border and not render.use_crop_to_border
        if render_border:
            start_y, end_y, start_x, end_x = render_border
        else: 
            size_x, size_y, start_x, end_x, start_y, end_y = scene_utils.get_render_borders(render, self.height, self.width)
        if use_border:            
            denoisedBeauty = denoisedBeauty[start_y:end_y,start_x:end_x,:]   
            denoisedAlpha = denoisedAlpha[start_y:end_y,start_x:end_x,:]   
        
        denoisedBeauty = denoisedBeauty.reshape((end_y-start_y)*(end_x-start_x), 3)
        denoisedAlpha = denoisedAlpha.reshape((end_y-start_y)*(end_x-start_x), 3)
        
        combined = np.concatenate((denoisedBeauty, denoisedAlpha), axis=1)
        combined = combined[:,:4] # alpha is 3 elements, we just want the first element
        denoised_passes["beauty"] = combined

        finished += 1
        self.stats_mgr._progress = int(100*finished/total)
        self.stats_mgr.draw_message("Denoising (beauty)")

        # now, denoise the other passes if we can
        for i, dspy_nm in enumerate(passes.keys()):
            if i == 0:
                continue
            p = passes[dspy_nm]
            self.stats_mgr.draw_message("Denoising (%s)" % dspy_nm)
            denoise_pass = None
            # Build features dict for this pass
            current_features = {
                'albedo': passAlbedo,
                'albedoVariance': passAlbedoVariance,
                'normal': passNormal,
                'normalVariance': passNormalVariance,
                'sampleCount': passSampleCount
            }
            if p["num_channels"] == 3:
                if p["pass_type"] == "color":
                    current_features["data"] = passInput
                    current_features["dataVariance"] = passInputVariance
                else:
                    current_features["data"] = passAlpha
                    current_features["dataVariance"] = passAlphaVariance
                if "asym" in self.topology and current_features["data"] is not None:
                    asymmetry = np.broadcast_to(self.asymmetry, shape=current_features["data"].shape[:-1] + (1,))
                    current_features["asymmetry"] = asymmetry
                self.denoiser.setFeatures(current_features, 0)
                self.denoiser.computeWeights()
                denoise_pass = self.denoiser.applyWeights([p['input']])
                if use_border:
                    denoise_pass = denoise_pass[start_y:end_y, start_x:end_x, :]
                denoise_pass = denoise_pass.reshape((end_y-start_y)*(end_x-start_x), 3)
            elif p["num_channels"] == 1:
                current_features["data"] = passAlpha
                current_features["dataVariance"] = passAlphaVariance
                if "asym" in self.topology and current_features["data"] is not None:
                    asymmetry = np.broadcast_to(self.asymmetry, shape=current_features["data"].shape[:-1] + (1,))
                    current_features["asymmetry"] = asymmetry
                self.denoiser.setFeatures(current_features, 0)
                self.denoiser.computeWeights()
                denoise_pass = self.denoiser.applyWeights([p['input']])  
                if use_border:
                    denoise_pass = denoise_pass[start_y:end_y, start_x:end_x, :]  
                denoise_pass = denoise_pass.reshape((end_y-start_y)*(end_x-start_x), 3)
                denoise_pass = denoise_pass[:, :1]  
            denoised_passes[dspy_nm] = denoise_pass   
            self.stats_mgr._progress = int(100 * (i + 1) / total)

        # Set progress to 100% at the end
        self.stats_mgr._progress = 100
        return denoised_passes
