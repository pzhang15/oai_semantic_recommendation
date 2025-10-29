import { create } from 'zustand'

type State = {
  lastQuery: string
  set: (p: Partial<State>) => void
}

export const useSearchStore = create<State>((set, get) => ({
  lastQuery: localStorage.getItem('q') || '',
  set: (p) => {
    const s = { ...get(), ...p }
    localStorage.setItem('q', s.lastQuery)
    set(p)
  }
}))


