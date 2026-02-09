# Pixxel Age Filter

Skin age estimation filter using the camera and MediaPipe Face Mesh. RTL Persian UI.

## Tech stack

- Vite, TypeScript, React
- shadcn-ui, Tailwind CSS
- @mediapipe/face_mesh, @mediapipe/camera_utils

## Local development

```bash
npm i
npm run dev
```

Open http://localhost:8080 (or the port Vite prints). Allow camera access.

## Build & run locally

```bash
npm run build
npm start
```

Serves the built app on port 3000 (or `PORT` if set).

## Deploy on Railway

1. Push this repo to GitHub.
2. In [Railway](https://railway.app), create a new project → **Deploy from GitHub repo** and select this repo.
3. Railway will:
   - Install dependencies
   - Run `npm run build`
   - Run `npm start` (serves `dist` on `PORT`)
4. Add a domain in the service **Settings → Networking**.

No env vars required. The app needs HTTPS and camera access in the browser.

## License

Private.
