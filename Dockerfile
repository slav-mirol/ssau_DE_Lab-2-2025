FROM n8nio/n8n

USER root

# Установка curl и ffmpeg
RUN apk update && \
    apk add --no-cache curl ffmpeg && \
    rm -rf /var/cache/apk/*

RUN npm install axios form-data

USER node