from setuptools import setup, find_packages

setup(
    name="werewolf",
    version="0.1",
    description="A project for werewolf game.",
    keywords="werewolf, gym",
    packages=find_packages(),
    install_requires=[
        "gym==0.26.2",
        "numpy>=1.24,<3.0",
        "openai>=1.59.3",
        "pydantic>=2.10.4",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.2",
        "tenacity>=9.0.0",
        "tiktoken>=0.7.0",
    ],
    extras_require={
        "strategy": [
            "torch>=2.0.0",
        ],
        "local_model": [
            "torch>=2.0.0",
            "transformers>=4.47.1",
        ],
        "vllm_server": [
            "vllm>=0.6.3",
        ],
    },
)
