FROM node:20-alpine AS build

WORKDIR /app/frontend

ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend/package.json ./
RUN npm install

COPY frontend ./
RUN npm run build

FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

