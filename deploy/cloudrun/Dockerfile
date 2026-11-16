# The HumansRUS checker on Google Cloud Run.
# Built from the folder that build_checker.sh makes; see deploy/README.md.
# Cloud Run sets PORT itself; 7860 is only the default for running the image elsewhere.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch (the default download includes GPU libraries and is several GB)
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# download the adopted model's vision backbone now, so the first check doesn't wait for it
RUN python -c "import json, torch; from shared import model_interface as mi; ck = torch.load(json.load(open('evaluation/results/threshold.json'))['checkpoint'], map_location='cpu', weights_only=False); print('backbone:', ck['backbone']); mi.load_backbone(ck['backbone'])"

EXPOSE 7860
CMD ["python", "product/web/serve.py"]
