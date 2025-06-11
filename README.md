---
language:
- en
library_name: stable-audio-tools
license: other
license_name: stable-audio-community
license_link: LICENSE
pipeline_tag: text-to-audio
extra_gated_prompt: By clicking "Agree", you agree to the [License Agreement](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md)
  and acknowledge Stability AI's [Privacy Policy](https://stability.ai/privacy-policy).
extra_gated_fields:
  Name: text
  Email: text
  Country: country
  Organization or Affiliation: text
  Receive email updates and promotions on Stability AI products, services, and research?:
    type: select
    options:
      - 'Yes'
      - 'No'
  What do you intend to use the model for?:
    type: select
    options:
      - Research
      - Personal use
      - Creative Professional
      - Startup
      - Enterprise
  I agree to the License Agreement and acknowledge Stability AI's Privacy Policy: checkbox
---

# Stable Audio Open 1.0

![Stable Audio Open logo](./stable_audio_light.png)

Please note: For commercial use, please refer to [https://stability.ai/license](https://stability.ai/license)

## Model Description
`Stable Audio Open 1.0` generates variable-length (up to 47s) stereo audio at 44.1kHz from text prompts. It comprises three components: an autoencoder that compresses waveforms into a manageable sequence length, a T5-based text embedding for text conditioning, and a transformer-based diffusion (DiT) model that operates in the latent space of the autoencoder.

## Usage

This model can be used with:
1. the [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) library
2. the [`diffusers`](https://huggingface.co/docs/diffusers/main/en/index) library


### Using with `stable-audio-tools`

This model is made to be used with the [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools) library for inference, for example:

```python
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    "prompt": "128 BPM tech house drum loop",
    "seconds_start": 0, 
    "seconds_total": 30
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
```

## Using with `diffusers`

Make sure you upgrade to the latest version of diffusers: `pip install -U diffusers`. And then you can run:

```py
import torch
import soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# define the prompts
prompt = "The sound of a hammer hitting a wooden surface."
negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("hammer.wav", output, pipe.vae.sampling_rate)

```
Refer to the [documentation](https://huggingface.co/docs/diffusers/main/en/index) for more details on optimization and usage.

### Fine-tuning

This repository includes a minimal example for fine-tuning the model using `diffusers`.
Install the required dependencies and run the script:

```bash
pip install torch diffusers transformers torchaudio
python train_finetune.py
```

The script expects a `train_dataset.csv` file with columns `path` and `prompt`
pointing to your audio files and their corresponding text descriptions. The
pretrained weights are loaded from `model_config.json`, and the script saves the
fine-tuned model in the `finetuned_model/` directory.




## Model Details
* **Model type**: `Stable Audio Open 1.0` is a latent diffusion model based on a transformer architecture.
* **Language(s)**: English
* **License**: [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md).
* **Commercial License**: to use this model commercially, please refer to [https://stability.ai/license](https://stability.ai/license)
* **Research Paper**: [https://arxiv.org/abs/2407.14358](https://arxiv.org/abs/2407.14358)

## Training dataset

### Datasets Used
Our dataset consists of 486492 audio recordings, where 472618 are from Freesound and 13874 are from the Free Music Archive (FMA). All audio files are licensed under CC0, CC BY, or CC Sampling+. This data is used to train our autoencoder and DiT. We use a publicly available pre-trained T5 model ([t5-base](https://huggingface.co/google-t5/t5-base)) for text conditioning.

### Attribution
Attribution for all audio recordings used to train Stable Audio Open 1.0 can be found on our [attribution page](https://info.stability.ai/attributions).

### Mitigations
We conducted an in-depth analysis to ensure no unauthorized copyrighted music was present in our training data before we began training.

To that end, we first identified music samples in Freesound using the [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) music classifier based on AudioSet classes. The identified music samples had at least 30 seconds of music that was predicted to belong to a music-related class with a threshold of 0.15 (PANNs output probabilities range from 0 to 1). This threshold was determined by classifying known music examples from FMA and ensuring no false negatives were present. 

The identified music samples were sent to Audible Magic’s identification services, a trusted content detection company, to ensure the absence of copyrighted music. Audible Magic flagged suspected copyrighted music, which we subsequently removed before training on the dataset. The majority of the removed content was field recordings in which copyrighted music was playing in the background. Following this procedure, we were left with 266324 CC0, 194840 CC-BY, and 11454 CC Sampling+ audio recordings.

We also conducted an in-depth analysis to ensure no copyrighted content was present in FMA's subset. In this case, the procedure was slightly different because the FMA subset consists of music signals. We did a metadata search against a large database of copyrighted music (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and flagged any potential match. The flagged content was reviewed individually by humans. After this process, we ended up with 8967 CC-BY and 4907 CC0 tracks.


## Use and Limitations


### Intended Use
The primary use of Stable Audio Open is research and experimentation on AI-based music and audio generation, including:

- Research efforts to better understand the limitations of generative models and further improve the state of science.
- Generation of music and audio guided by text to explore current abilities of generative AI models by machine learning practitioners and artists.


### Out-of-Scope Use Cases
The model should not be used on downstream applications without further risk evaluation and mitigation. The model should not be used to intentionally create or disseminate audio or music pieces that create hostile or alienating environments for people.


### Limitations
- The model is not able to generate realistic vocals.
- The model has been trained with English descriptions and will not perform as well in other languages.
- The model does not perform equally well for all music styles and cultures.
- The model is better at generating sound effects and field recordings than music.
- It is sometimes difficult to assess what types of text descriptions provide the best generations. Prompt engineering may be required to obtain satisfying results.


### Biases
The source of data is potentially lacking diversity and all cultures are not equally represented in the dataset. The model may not perform equally well on the wide variety of music genres and sound effects that exist. The generated samples from the model will reflect the biases from the training data.