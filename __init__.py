import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

ckpt_dir = os.path.join(now_dir,"checkpoints")
seedvc_dir = os.path.join(ckpt_dir,"Seed-VC")
facodec_dir = os.path.join(ckpt_dir,"FAcodec")
campplus_dir = os.path.join(ckpt_dir, "campplus")

import torch
import yaml
import time
import torchaudio
from huggingface_hub import snapshot_download
import torchaudio.compliance.kaldi as kaldi
from seedvc.modules.commons import recursive_munch,build_model,load_checkpoint           

class SeedVCNode:

    def __init__(self):
        if not os.path.exists(os.path.join(seedvc_dir,"DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth")):
            snapshot_download(repo_id="Plachta/Seed-VC",local_dir=seedvc_dir)
        if not os.path.exists(os.path.join(facodec_dir,"pytorch_model.bin")):
            snapshot_download(repo_id="Plachta/FAcodec",local_dir=facodec_dir)

        if not os.path.exists(os.path.join(campplus_dir,"campplus_cn_common.bin")):
            snapshot_download(repo_id="funasr/campplus",local_dir=campplus_dir)

        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source":("AUDIO",),
                "target":("AUDIO",),
                "diffusion_steps":("INT",{
                    "default": 10
                }),
                "length_adjust":("FLOAT",{
                    "default": 1.0
                }),
                "inference_cfg_rate":("FLOAT",{
                    "default": 0.7
                }),
                "n_quantizers":("INT",{
                    "default": 3
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_SeedVC"

    def cfy2librosa(self,audio,target_sr):
        waveform = audio["waveform"].squeeze(0)
        sr = audio["sample_rate"]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        print(waveform.shape)
        print(f"from {sr} to {target_sr}")
        return waveform.numpy()[0]

    def gen_audio(self,source,target,diffusion_steps,length_adjust,
                  inference_cfg_rate,n_quantizers):
        # Load model and configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dit_config_path = os.path.join(seedvc_dir,"config_dit_mel_seed_facodec_small_wavenet.yml")
        if self.model is None:
            dit_config_path = os.path.join(seedvc_dir,"config_dit_mel_seed_wavenet.yml")
            config = yaml.safe_load(open(dit_config_path, 'r'))
            model_params = recursive_munch(config['model_params'])
            model = build_model(model_params, stage='DiT')
            hop_length = config['preprocess_params']['spect_params']['hop_length']
            self.sr = config['preprocess_params']['sr']

            # Load checkpoints
            dit_checkpoint_path = os.path.join(seedvc_dir,"DiT_step_315000_seed_v2_wavenet_online_pruned.pth")
            # dit_checkpoint_path = os.path.join(seedvc_dir,"DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth")
            self.model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                            load_only_params=True, ignore_modules=[], is_distributed=False)
            for key in self.model:
                self.model[key].eval()
                self.model[key].to(device)
            self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

            # Load additional modules
            from seedvc.modules.campplus.DTDNN import CAMPPlus

            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(torch.load(os.path.join(campplus_dir,"campplus_cn_common.bin"), map_location='cpu'))
            self.campplus_model.eval()
            self.campplus_model.to(device)

            from seedvc.modules.hifigan.generator import HiFTGenerator
            from seedvc.modules.hifigan.f0_predictor import ConvRNNF0Predictor

            hift_checkpoint_path = os.path.join(seedvc_dir,"hift.pt")
            hift_config_path = os.path.join(seedvc_dir,"hifigan.yml")
                                                            
            hift_config = yaml.safe_load(open(hift_config_path, 'r'))
            self.hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            self.hift_gen.load_state_dict(torch.load(hift_checkpoint_path, map_location='cpu'))
            self.hift_gen.eval()
            self.hift_gen.to(device)

            from seedvc.modules.cosyvoice_tokenizer.frontend import CosyVoiceFrontEnd
            speech_tokenizer_path = os.path.join(seedvc_dir, "speech_tokenizer_v1.onnx")
            self.cosyvoice_frontend = CosyVoiceFrontEnd(speech_tokenizer_model=speech_tokenizer_path,
                                                device='cuda', device_id=0)
            
            # Generate mel spectrograms
            mel_fn_args = {
                "n_fft": config['preprocess_params']['spect_params']['n_fft'],
                "win_size": config['preprocess_params']['spect_params']['win_length'],
                "hop_size": config['preprocess_params']['spect_params']['hop_length'],
                "num_mels": config['preprocess_params']['spect_params']['n_mels'],
                "sampling_rate": self.sr,
                "fmin": 0,
                "fmax": 8000,
                "center": False
            }
            from seedvc.modules.audio import mel_spectrogram

            self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        sr = self.sr
        source_audio = self.cfy2librosa(source,sr)
        ref_audio = self.cfy2librosa(target,sr)

        source_audio = source_audio[:sr * 30]
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

        ref_audio = ref_audio[:(sr * 30 - source_audio.size(-1))]
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

        source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

        print(f"from {sr} to 16000")
        with torch.no_grad():
            S_alt = [
                self.cosyvoice_frontend.extract_speech_token(source_waves_16k, )
            ]
            S_alt_lens = torch.LongTensor([s[1] for s in S_alt]).to(device)
            S_alt = torch.cat([torch.nn.functional.pad(s[0], (0, max(S_alt_lens) - s[0].size(1))) for s in S_alt], dim=0)

            S_ori = [
                self.cosyvoice_frontend.extract_speech_token(ref_waves_16k, )
            ]
            S_ori_lens = torch.LongTensor([s[1] for s in S_ori]).to(device)
            S_ori = torch.cat([torch.nn.functional.pad(s[0], (0, max(S_ori_lens) - s[0].size(1))) for s in S_ori], dim=0)

            mel = self.to_mel(source_audio.to(device).float())
            mel2 = self.to_mel(ref_audio.to(device).float())

            target = mel
            target2 = mel2

            target_lengths = torch.LongTensor([int(target.size(2) * length_adjust)]).to(target.device)
            target2_lengths = torch.LongTensor([target2.size(2)]).to(target2.device)

            feat2 = kaldi.fbank(ref_waves_16k,
                                num_mel_bins=80,
                                dither=0,
                                sample_frequency=16000)
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))

            # print(target_lengths.shape)
            cond = self.model.length_regulator(S_alt, ylens=target_lengths)[0]
            prompt_condition = self.model.length_regulator(S_ori, ylens=target2_lengths)[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            prompt_target = target2

            time_vc_start = time.time()
            vc_target = self.model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(prompt_target.device), prompt_target, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, prompt_target.size(-1):]
            vc_wave = self.hift_gen.inference(vc_target)
            time_vc_end = time.time()
            print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

        res_audio = {
            "waveform": vc_wave.cpu().unsqueeze(0),
            "sample_rate": sr
        }
        return (res_audio, )

NODE_CLASS_MAPPINGS = {
    "SeedVCNode": SeedVCNode
}






        