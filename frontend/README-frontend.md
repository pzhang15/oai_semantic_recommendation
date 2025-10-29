# Frontend (React + Vite)

## Install
```bash
npm i
```

## Env
Create a `.env` with:
```bash
VITE_API_URL=http://localhost:8000
```

## Run
```bash
npm run dev
```

## Build
```bash
npm run build
```

Routes:
- /search
- /outfit
- /diagnostics

Notes:
- Uses Tailwind and minimal components; you can add shadcn/ui via their CLI later if desired.
- TanStack Query handles fetch + caching.
- Basic zod validation guards server responses.
