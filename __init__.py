import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(now_dir)

ckpt_dir = os.path.join(now_dir,"checkpoints")
seedvc_dir = os.path.join(ckpt_dir,"Seed-VC")
facodec_dir = os.path.join(ckpt_dir,"FAcodec")
campplus_dir = os.path.join(ckpt_dir, "campplus")
rmvpe_dir = os.path.join(ckpt_dir,"rmvpe")
bigvgan_dir = os.path.join(ckpt_dir,"bigvgan")

import torch
import yaml
import time
import traceback
import torchaudio
from huggingface_hub import snapshot_download,hf_hub_download
import torchaudio.compliance.kaldi as kaldi
from slicer2 import Slicer
import numpy as np
from seedvc.modules.rmvpe import RMVPE
from seedvc.modules.bigvgan import bigvgan
from seedvc.modules.campplus.DTDNN import CAMPPlus
from seedvc.modules.audio import mel_spectrogram
from seedvc.modules.hifigan.generator import HiFTGenerator
from seedvc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
from seedvc.modules.commons import recursive_munch,build_model,load_checkpoint           
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_models():
    if not os.path.exists(os.path.join(seedvc_dir,"DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth")):
            snapshot_download(repo_id="Plachta/Seed-VC",local_dir=seedvc_dir)
    if not os.path.exists(os.path.join(facodec_dir,"pytorch_model.bin")):
        snapshot_download(repo_id="Plachta/FAcodec",local_dir=facodec_dir)

    if not os.path.exists(os.path.join(campplus_dir,"campplus_cn_common.bin")):
        snapshot_download(repo_id="funasr/campplus",local_dir=campplus_dir)

    if not os.path.exists(os.path.join(rmvpe_dir,"rmvpe.pt")):
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI",filename="rmvpe.pt",
                        local_dir=rmvpe_dir)
    if not os.path.exists(os.path.join(bigvgan_dir,"bigvgan_generator.pt")):
        snapshot_download(repo_id="nvidia/bigvgan_v2_22khz_80band_256x",local_dir=bigvgan_dir,
                          allow_patterns=["*.json","*generator.pt"])

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

class AudioSlicerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "threshold":("INT",{
                    "default": -40,
                }),
                "min_length":("INT",{
                    "default": 5000,
                }),
                "min_interval":("INT",{
                    "default": 300,
                }),
                "hop_size":("INT",{
                    "default": 20,
                }),
                "max_sil_kept":("INT",{
                    "default": 5000,
                }),
            }
        }
    RETURN_TYPES = ("SLICER",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "slice"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_SeedVC"

    def slice(self,threshold,min_length,min_interval,hop_size,max_sil_kept):
        slicer = Slicer(
            threshold= threshold,
            min_length= min_length,
            min_interval= min_interval,
            hop_size= hop_size,
            max_sil_kept= max_sil_kept
        )
        return (slicer,)
    
'''
class SeedVC4SingNode:
    def __init__(self): 
        self.model = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source":("AUDIO",),
                "target":("AUDIO",),
                "semi_tone_shift":("INT",{
                    "default": 0
                }),
                "diffusion_steps":("INT",{
                    "default": 25
                }),
                "length_adjust":("FLOAT",{
                    "default": 1.0
                }),
                "inference_cfg_rate":("FLOAT",{
                    "default": 0.7
                }),
                "n_quantizers":("INT",{
                    "default": 3
                })
            },
            "optional":{
                "slicer":("SLICER",)
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

    def get_single_audio(self,source_audio,ref_audio,diffusion_steps,
                         inference_cfg_rate,semi_tone_shift,n_quantizers,
                         length_adjust=1.0):
        sr = self.sr
        source_audio = source_audio[:sr * 30]
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

        ref_audio = ref_audio[:sr*30]
        ref_audio = ref_audio[:(sr * 30 - source_audio.size(-1))]
        ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

        source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

        with torch.no_grad():
            converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
            wave_lengths_24k = torch.LongTensor([converted_waves_24k.size(1)]).to(converted_waves_24k.device)
            waves_input = converted_waves_24k.unsqueeze(1)
            z = self.codec_encoder.encoder(waves_input)
            (
                quantized,
                codes
            ) = self.codec_encoder.quantizer(
                z,
                waves_input,
            )
            S_alt = torch.cat([codes[1], codes[0]], dim=1)

            # S_ori should be extracted in the same way
            waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
            waves_input = waves_24k.unsqueeze(1)
            z = self.codec_encoder.encoder(waves_input)
            (
                quantized,
                codes
            ) = self.codec_encoder.quantizer(
                z,
                waves_input,
            )
            S_ori = torch.cat([codes[1], codes[0]], dim=1)
            
            mel = self.to_mel(source_audio.to(device).float())
            mel2 = self.to_mel(ref_audio.to(device).float())

            target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))
            
            ## f0
            waves_16k = torchaudio.functional.resample(waves_24k, sr, 16000)
            converted_waves_16k = torchaudio.functional.resample(converted_waves_24k, sr, 16000)
            F0_ori = self.rmvpe.infer_from_audio(waves_16k[0], thred=0.03)
            F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

            F0_ori = torch.from_numpy(F0_ori).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)
            # mean_log_f0_ori = torch.mean(voiced_log_f0_ori)
            # mean_log_f0_alt = torch.mean(voiced_log_f0_alt)

            # shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if semi_tone_shift != 0:
                shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], semi_tone_shift)


            # Length regulation
            cond = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=int(n_quantizers), f0=shifted_f0_alt)[0]
            prompt_condition = self.model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers), f0=F0_ori)[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)

            

            
            vc_target = self.model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device), mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = self.hift_gen.inference(vc_target)
        return vc_wave
    
    def gen_audio(self,source,target,semi_tone_shift,diffusion_steps,
                  length_adjust,inference_cfg_rate,n_quantizers,slicer=None):

        dit_config_path = os.path.join(seedvc_dir,"config_dit_mel_seed_facodec_small_wavenet_f0.yml")
        dit_checkpoint_path = os.path.join(seedvc_dir,"DiT_step_404000_seed_v2_uvit_facodec_small_wavenet_f0_pruned.pth")
        
        if self.model is None:
            config_path = os.path.join(facodec_dir,"config.yml")
            ckpt_path = os.path.join(facodec_dir, 'pytorch_model.bin')

            codec_config = yaml.safe_load(open(config_path,"r"))
            codec_model_params = recursive_munch(codec_config['model_params'])
            codec_encoder = build_model(codec_model_params, stage="codec")

            ckpt_params = torch.load(ckpt_path, map_location="cpu")

            for key in codec_encoder:
                codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
            _ = [codec_encoder[key].eval() for key in codec_encoder]
            _ = [codec_encoder[key].to(device) for key in codec_encoder]
            self.codec_encoder = codec_encoder

            # f0 conditioned model
            config = yaml.safe_load(open(dit_config_path, 'r'))
            model_params = recursive_munch(config['model_params'])
            model_f0 = build_model(model_params, stage='DiT')
            hop_length = config['preprocess_params']['spect_params']['hop_length']
            self.sr = config['preprocess_params']['sr']

            # Load checkpoints
            model_f0, _, _, _ = load_checkpoint(model_f0, None, dit_checkpoint_path,
                                            load_only_params=True, ignore_modules=[], is_distributed=False)
            for key in model_f0:
                model_f0[key].eval()
                model_f0[key].to(device)
            model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.model = model_f0
            # f0 extractor
            from seedvc.modules.rmvpe import RMVPE

            self.rmvpe = RMVPE(os.path.join(rmvpe_dir,"rmvpe.pt"), is_half=False, device=device)

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

        if slicer:
            slicer.load_params(sr)
            waves = []
            chunks = slicer.slice(source_audio)
            
            for i, (chunk, start, end) in enumerate(chunks):
                # tmp_max = np.abs(chunk).max()
                # if(tmp_max>1):chunk/=tmp_max
                # chunk = (chunk / tmp_max * (0.9 * 0.25)) + (1 - 0.25) * chunk
                if i == 0:
                    silent_wave = np.zeros((1,start))
                    waves.append(silent_wave)
                    print(silent_wave.shape)
                time_vc_start = time.time()
                chunk_duration = (end-start)//sr
                try:
                    fake_wave = self.get_single_audio(source_audio[start:end],ref_audio,diffusion_steps,
                                                    inference_cfg_rate,semi_tone_shift,n_quantizers)
                    time_vc_end = time.time()
                    print(f"i {i} chunk duration:{chunk_duration} RTF: {(time_vc_end - time_vc_start) / fake_wave.size(-1) * sr}")
                    waves.append(fake_wave.cpu().numpy())
                    dur = end-start-fake_wave.size(-1)
                    if dur > 0:
                        silent_wave = np.zeros((1,dur))
                        waves.append(silent_wave)
                except:
                    print(f"{chunk_duration} > 30s can't get fake, slient replace")
                    torchaudio.save(f"{i}_temp.wav",torch.tensor(source_audio[start:end]).unsqueeze(0),sr)
                    silent_wave = np.zeros((1,end-start))
                    waves.append(silent_wave)
                    # traceback.print_exc()

                if i == len(chunks) - 1 and source_audio.shape[0]>end:
                    silent_wave = np.zeros((1,source_audio.shape[0]-end))
                    waves.append(silent_wave)

            vc_wave = np.concatenate(waves,axis=-1)
            # print(vc_wave.shape)
            # assert vc_wave.shape[-1] == source_audio.shape[0], f"{vc_wave.shape} not = {source_audio.shape}"
            res_audio = {
                "waveform": torch.FloatTensor(vc_wave).unsqueeze(0),
                "sample_rate": sr
            }
        else:
            time_vc_start = time.time()
            vc_wave = self.get_single_audio(source_audio,ref_audio,diffusion_steps,
                                        inference_cfg_rate,semi_tone_shift,n_quantizers,
                                        length_adjust)
            time_vc_end = time.time()
            print(f"RTF: {(time_vc_end - time_vc_start) / fake_wave.size(-1) * sr}")
                    
            res_audio = {
                "waveform": vc_wave.unsqueeze(0),
                "sample_rate": sr
            }
        return (res_audio, )
'''
def crossfade(chunk1, chunk2, overlap):
    fade_out = np.linspace(1, 0, overlap)
    fade_in = np.linspace(0, 1, overlap)
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

def load_models(if_f0=False):
    # Load model and configuration
    if not if_f0:
        dit_checkpoint_path =os.path.join(seedvc_dir,"DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth")
        dit_config_path =  os.path.join(seedvc_dir,"config_dit_mel_seed_facodec_small_wavenet.yml")

        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        model = build_model(model_params, stage='DiT')
        hop_length = config['preprocess_params']['spect_params']['hop_length']
        sr = config['preprocess_params']['sr']
        # Load checkpoints
        model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                        load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model:
            model[key].eval()
            model[key].to(device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        # Generate mel spectrograms
        mel_fn_args = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": 0,
            "fmax": 8000,
            "center": False
        }
        

        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
        

        hift_checkpoint_path = os.path.join(seedvc_dir,"hift.pt")
        hift_config_path = os.path.join(seedvc_dir,"hifigan.yml")
        hift_config = yaml.safe_load(open(hift_config_path, 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_gen.load_state_dict(torch.load(hift_checkpoint_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)

    else:
        # f0 conditioned model
        dit_checkpoint_path =os.path.join(seedvc_dir,"DiT_seed_v2_uvit_facodec_small_wavenet_f0_bigvgan_pruned.pth")
        dit_config_path =  os.path.join(seedvc_dir,"config_dit_mel_seed_facodec_small_wavenet_f0.yml")
        config = yaml.safe_load(open(dit_config_path, 'r'))
        model_params = recursive_munch(config['model_params'])
        model_f0 = build_model(model_params, stage='DiT')
        hop_length = config['preprocess_params']['spect_params']['hop_length']
        sr = config['preprocess_params']['sr']

        # Load checkpoints
        model_f0, _, _, _ = load_checkpoint(model_f0, None, dit_checkpoint_path,
                                        load_only_params=True, ignore_modules=[], is_distributed=False)
        for key in model_f0:
            model_f0[key].eval()
            model_f0[key].to(device)
        model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # f0 extractor

        model_path = os.path.join(rmvpe_dir,"rmvpe.pt")
        rmvpe = RMVPE(model_path, is_half=False, device=device)
        mel_fn_args_f0 = {
            "n_fft": config['preprocess_params']['spect_params']['n_fft'],
            "win_size": config['preprocess_params']['spect_params']['win_length'],
            "hop_size": config['preprocess_params']['spect_params']['hop_length'],
            "num_mels": config['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": sr,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        to_mel = lambda x: mel_spectrogram(x, **mel_fn_args_f0)
        

        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_dir, use_cuda_kernel=False)

        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)


    # Load additional modules

    campplus_ckpt_path = os.path.join(campplus_dir, "campplus_cn_common.bin")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    ## load 
    ckpt_path = os.path.join(facodec_dir,'pytorch_model.bin')
    config_path = os.path.join(facodec_dir,'config.yml')

    codec_config = yaml.safe_load(open(config_path))
    codec_model_params = recursive_munch(codec_config['model_params'])
    codec_encoder = build_model(codec_model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in codec_encoder:
        codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
    _ = [codec_encoder[key].eval() for key in codec_encoder]
    _ = [codec_encoder[key].to(device) for key in codec_encoder]

    if not if_f0:
        return model,sr,campplus_model,hift_gen,codec_encoder,to_mel,hop_length
    else:
        return model_f0,sr,campplus_model,bigvgan_model,codec_encoder,to_mel,rmvpe,hop_length

class SeedVC4SingNode:
    def __init__(self): 
        download_models()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "source":("AUDIO",),
                "target":("AUDIO",),
                "pitch_shift":("INT",{
                    "default": 0
                }),
                "diffusion_steps":("INT",{
                    "default": 50
                }),
                "length_adjust":("FLOAT",{
                    "default": 1.0
                }),
                "inference_cfg_rate":("FLOAT",{
                    "default": 0.7
                }),
                "n_quantizers":("INT",{
                    "default": 3
                })
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

    def gen_audio(self,source,target,pitch_shift,diffusion_steps,
                  length_adjust,inference_cfg_rate,n_quantizers):
        
        model,sr,campplus_model,bigvgan_model,codec_encoder,to_mel,rmvpe,hop_length = load_models(if_f0=True)
        # streaming and chunk processing related params
        max_context_window = sr // hop_length * 30
        overlap_frame_len = 64
        overlap_wave_len = overlap_frame_len * hop_length

        source_audio = self.cfy2librosa(source,sr)
        ref_audio = self.cfy2librosa(target,sr)
        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

        # Resample
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

        # Extract features
        converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
        waves_input = converted_waves_24k.unsqueeze(1)
        max_wave_len_per_chunk = 24000 * 20
        wave_input_chunks = [
            waves_input[..., i:i + max_wave_len_per_chunk] for i in range(0, waves_input.size(-1), max_wave_len_per_chunk)
        ]
        S_alt_chunks = []
        for i, chunk in enumerate(wave_input_chunks):
            z = codec_encoder.encoder(chunk)
            (
                quantized,
                codes
            ) = codec_encoder.quantizer(
                z,
                chunk,
            )
            S_alt = torch.cat([codes[1], codes[0]], dim=1)
            S_alt_chunks.append(S_alt)
        S_alt = torch.cat(S_alt_chunks, dim=-1)

        # S_ori should be extracted in the same way
        waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
        waves_input = waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (
            quantized,
            codes
        ) = codec_encoder.quantizer(
            z,
            waves_input,
        )
        S_ori = torch.cat([codes[1], codes[0]], dim=1)

        mel = to_mel(source_audio.to(device).float())
        mel2 = to_mel(ref_audio.to(device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                num_mel_bins=80,
                                                dither=0,
                                                sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = campplus_model(feat2.unsqueeze(0))

        # if f0_condition:
        waves_16k = torchaudio.functional.resample(waves_24k, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(converted_waves_24k, sr, 16000)
        F0_ori = rmvpe.infer_from_audio(waves_16k[0], thred=0.03)
        F0_alt = rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)
        # mean_log_f0_ori = torch.mean(voiced_log_f0_ori)
        # mean_log_f0_alt = torch.mean(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

        # Length regulation
        cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=int(n_quantizers), f0=shifted_f0_alt)[0]
        prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers), f0=F0_ori)[0]

        max_source_window = max_context_window - mel2.size(2)
        # split source condition (cond) into chunks
        processed_frames = 0
        generated_wave_chunks = []
        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            # Voice Conversion
            vc_target = model.cfm.inference(cat_condition,
                                                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                    mel2, style2, None, diffusion_steps,
                                                    inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = bigvgan_model(vc_target)[0]
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
        
        res_audio = {
                        "waveform": torch.FloatTensor(np.concatenate(generated_wave_chunks)).unsqueeze(0).unsqueeze(0),
                        "sample_rate": sr
                    }
        return (res_audio, )

class SeedVCNode:

    def __init__(self):
        download_models()

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
        model,sr,campplus_model,hift_gen,codec_encoder,to_mel,hop_length = load_models()
        # streaming and chunk processing related params
        max_context_window = sr // hop_length * 30
        overlap_frame_len = 64
        overlap_wave_len = overlap_frame_len * hop_length
        bitrate = "320k"

        source_audio = self.cfy2librosa(source,sr)
        ref_audio = self.cfy2librosa(target,sr)

        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

        # Resample
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

        
        with torch.no_grad():
            # Extract features
            converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
            waves_input = converted_waves_24k.unsqueeze(1)
            max_wave_len_per_chunk = 24000 * 20
            wave_input_chunks = [
                waves_input[..., i:i + max_wave_len_per_chunk] for i in range(0, waves_input.size(-1), max_wave_len_per_chunk)
            ]
            S_alt_chunks = []
            for i, chunk in enumerate(wave_input_chunks):
                z = codec_encoder.encoder(chunk)
                (
                    quantized,
                    codes
                ) = codec_encoder.quantizer(
                    z,
                    chunk,
                )
                S_alt = torch.cat([codes[1], codes[0]], dim=1)
                S_alt_chunks.append(S_alt)
            S_alt = torch.cat(S_alt_chunks, dim=-1)

            # S_ori should be extracted in the same way
            waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
            waves_input = waves_24k.unsqueeze(1)
            z = codec_encoder.encoder(waves_input)
            (
                quantized,
                codes
            ) = codec_encoder.quantizer(
                z,
                waves_input,
            )
            S_ori = torch.cat([codes[1], codes[0]], dim=1)

            mel = to_mel(source_audio.to(device).float())
            mel2 = to_mel(ref_audio.to(device).float())

            target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                                    num_mel_bins=80,
                                                    dither=0,
                                                    sample_frequency=16000)
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = campplus_model(feat2.unsqueeze(0))

            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None

            # Length regulation
            cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=int(n_quantizers), f0=shifted_f0_alt)[0]
            prompt_condition = model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers), f0=F0_ori)[0]

            max_source_window = max_context_window - mel2.size(2)
            # split source condition (cond) into chunks
            processed_frames = 0
            generated_wave_chunks = []
            # generate chunk by chunk and stream the output
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                # Voice Conversion
                vc_target = model.cfm.inference(cat_condition,
                                                        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                        mel2, style2, None, diffusion_steps,
                                                        inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1):]
                vc_wave = hift_gen.inference(vc_target, f0=None)
                if processed_frames == 0:
                    if is_last_chunk:
                        output_wave = vc_wave[0].cpu().numpy()
                        generated_wave_chunks.append(output_wave)
                        break
                    output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len
                elif is_last_chunk:
                    output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                    generated_wave_chunks.append(output_wave)
                    processed_frames += vc_target.size(2) - overlap_frame_len
                    break
                else:
                    output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len
        res_audio = {
                        "waveform": torch.FloatTensor(np.concatenate(generated_wave_chunks)).unsqueeze(0).unsqueeze(0),
                        "sample_rate": sr
                    }
        return (res_audio, )

NODE_CLASS_MAPPINGS = {
    "SeedVC4SingNode":SeedVC4SingNode,
    "SeedVCNode": SeedVCNode
}