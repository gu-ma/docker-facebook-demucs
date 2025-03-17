import gradio as gr
from scipy.io.wavfile import write
import subprocess
from pathlib import Path


def inference(audio, model, shift, overlap):
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)
    empty_path = Path("")
    input_file = output_dir / "test.wav"
    write(input_file, audio[0], audio[1])

    command = [
        "python3",
        "-m",
        "demucs.separate",
        "--shifts",
        f"{shift}",
        "--overlap",
        f"{overlap}",
        "-n",
        model,
        "-d",
        "cuda",
        str(input_file),
        "-o",
        str(output_dir),
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        print(f"Error in Demucs script: {process.stderr.decode()}")
        return [None] * 6  # Return six None values to match the expected output count

    print("Demucs script output:", process.stdout.decode())

    # Determine the number of output files based on the model
    if model == "htdemucs_6s":
        files = [
            output_dir / model / "test" / "vocals.wav",
            output_dir / model / "test" / "bass.wav",
            output_dir / model / "test" / "drums.wav",
            output_dir / model / "test" / "other.wav",
            output_dir / model / "test" / "piano.wav",
            output_dir / model / "test" / "guitar.wav",
        ]
    else:
        files = [
            output_dir / model / "test" / "vocals.wav",
            output_dir / model / "test" / "bass.wav",
            output_dir / model / "test" / "drums.wav",
            output_dir / model / "test" / "other.wav",
            empty_path,
            empty_path,
        ]

    existing_files = [file if file and file.is_file() else None for file in files]
    if not any(existing_files):
        print("No output files found.")

    return existing_files


title = "Demucs"
model_info = """
The list of pre-trained models is:
- `htdemucs`: first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
- `htdemucs_ft`: fine-tuned version of `htdemucs`, separation will take 4 times more time
    but might be a bit better. Same training set as `htdemucs`.
- `htdemucs_6s`: 6 sources version of `htdemucs`, with `piano` and `guitar` being added as sources.
    Note that the `piano` source is not working great at the moment.
- `hdemucs_mmi`: Hybrid Demucs v3, retrained on MusDB + 800 songs.
- `mdx`: trained only on MusDB HQ, winning model on track A at the [MDX][mdx] challenge.
- `mdx_extra`: trained with extra training data (**including MusDB test set**), ranked 2nd on the track B
    of the [MDX][mdx] challenge.
- `mdx_q`, `mdx_extra_q`: quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
"""
slider_info = """
The `shifts` performs multiple predictions with random shifts (a.k.a the *shift trick*) of the input and average them. This makes prediction `SHIFTS` times slower. Don't use it unless you have a GPU.  
The `overlap` option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine.It can probably be reduced to 0.1 to improve a bit speed.
"""

description = f"## Models\n{model_info}\n## Options\n{slider_info}\n[Documentation](https://github.com/adefossez/demucs)"
article = (
    "<p style='text-align: center'><a href='https://arxiv.org/abs/1911.13254' target='_blank'>Music Source Separation in the Waveform Domain</a> | "
    "<a href='https://github.com/adefossez/demucs' target='_blank'>Github Repo</a></p>"
)
examples = [["test.mp3"]]
model_choices = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_extra",
    "mdx_q",
    "mdx_extra_q",
]  # Complete list of available model choices

gr.Interface(
    inference,
    [
        gr.components.Audio(type="numpy", label="Input"),
        gr.components.Dropdown(
            choices=model_choices, label="Model", value="mdx_extra_q"
        ),
        gr.components.Slider(minimum=1, maximum=10, step=1, label="Shifts", value=1),
        gr.components.Slider(
            minimum=0.0, maximum=1.0, step=0.01, label="Overlap", value=0.25
        ),
    ],
    [
        gr.components.Audio(type="filepath", label="Vocals"),
        gr.components.Audio(type="filepath", label="Bass"),
        gr.components.Audio(type="filepath", label="Drums"),
        gr.components.Audio(type="filepath", label="Other"),
        gr.components.Audio(type="filepath", label="Piano"),
        gr.components.Audio(type="filepath", label="Guitar"),
    ],
    title=title,
    description=description,
    article=article,
).launch(
    server_name="0.0.0.0",
)
