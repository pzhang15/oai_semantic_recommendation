import { Outlet, Link, useLocation } from 'react-router-dom'

export default function App() {
  const loc = useLocation()
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-white">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-4">
          <Link to="/" className="font-semibold">Semantic Fashion Recs</Link>
          <nav className="flex items-center gap-4 text-sm">
            <Link className={navClass(loc.pathname==='/search'||loc.pathname==='/')} to="/search">Search</Link>
            <Link className={navClass(loc.pathname==='/diagnostics')} to="/diagnostics">Diagnostics</Link>
          </nav>
          <div className="ml-auto" />
        </div>
      </header>
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t text-xs text-neutral-500">
        <div className="max-w-6xl mx-auto px-4 py-3">© {new Date().getFullYear()} Semantic Recs</div>
      </footer>
    </div>
  )
}

function navClass(active: boolean) {
  return active ? 'underline underline-offset-4' : 'hover:underline'
}


