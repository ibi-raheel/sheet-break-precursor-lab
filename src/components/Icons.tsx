export const Chevron = ({ dir }: { dir: 'left' | 'right' }) => (
  <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    {dir === 'left' ? <path d="M8.5 3 4.5 7l4 4" /> : <path d="M5.5 3l4 4-4 4" />}
  </svg>
)
