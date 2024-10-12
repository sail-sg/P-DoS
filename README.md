# Denial-of-Service Poisoning Attacks on Large Language Models

## Installation

This code is tested on our local environment (python=3.10.12, cuda=11.8), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n pdos python=3.10.12
```
Then, activate the environment:
```bash
conda activate pdos
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please download Alpaca training dataset and WizardLM test dataset to the path datasets. In addition, download LLaMA-2-Chat-7B in /your_llama2_chat_hf_path.

## P-DoS attacks for LLMs by Data Contributors

Run the following command to launch P-DoS attacks for GPT-4o.

```shell
python pdos.py
```

## P-DoS attacks for LLMs by Model Publishers

Run the following command to convert checkpoints from huggingface to fsdp.

```shell
bash scripts/convert.sh
```

Run the following command to launch P-DoS (CSF).

```shell
bash scripts/dos_csf.sh
```

Run the following command to launch P-DoS (L_DoS).

```shell
bash scripts/dos_loss.sh
```

Run the following command to evaluate DoS attacks for LLMs.

```shell
bash scripts/eval.sh
```

## Citation

```
@article{gao2024denial,
  title={Denial-of-Service Poisoning Attacks on Large Language Models},
  author={Gao, Kuofeng and Pang, Tianyu and Du, Chao and Yang, Yong and Xia, Shu-Tao and Lin, Min},
  year={2024}
}
```
