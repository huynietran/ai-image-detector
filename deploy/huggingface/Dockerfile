# The HumansRUS checker on a Hugging Face Space (Docker, free CPU).
# Built from the folder that build_space.sh makes; see deploy/README.md.
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

# download DINOv2 now so the first check doesn't wait for it
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "product/web/serve.py"]
