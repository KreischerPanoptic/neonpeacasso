from setuptools import setup, find_packages

setup(
    name='neonpeacasso',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/neonsecret/neonpeacasso',
    license_files=('LICENSE',),
    author='neon',
    author_email='slava.v.ivannikov@gmail.com',
    description='UI tool to help you generate art (and experiment) with multimodal (text, image) AI models (stable '
                'diffusion) ',
    install_requires=[
        "transformers",
        "scipy",
        "ftfy",
        "diffusers==0.2.4",
        "torch",
        "pydantic",
        "uvicorn",
        "typer",
        "fastapi",
        "albumentations",
        "opencv-python",
        "pudb",
        "imageio",
        "imageio-ffmpeg",
        "pytorch-lightning",
        "omegaconf",
        "test-tube",
        "streamlit>=0.73.1",
        "einops",
        "torch-fidelity",
        "transformers",
        "torchmetrics",
        "kornia",
        "taming-transformers @ git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        "CLIP @ git+https://github.com/openai/CLIP.git@main#egg=clip",
        "latent-diffusion @ git+https://github.com/neonsecret/stable-diffusion",
    ],
    entry_points={
        "console_scripts": ['neonpeacasso = neonpeacasso.cli:run']
    },
)
