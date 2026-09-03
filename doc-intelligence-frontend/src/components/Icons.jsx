const paths = {
  archive: <><path d="M4 7.5h16v12H4z"/><path d="M3 4.5h18v3H3zM9 11h6"/></>,
  arrowRight: <><path d="M5 12h14M14 7l5 5-5 5"/></>,
  bookOpen: <><path d="M3.5 5.5c3.2-.7 6.1 0 8.5 2v12c-2.4-2-5.3-2.7-8.5-2z"/><path d="M20.5 5.5c-3.2-.7-6.1 0-8.5 2v12c2.4-2 5.3-2.7 8.5-2z"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  chevronDown: <path d="m7 10 5 5 5-5"/>,
  close: <><path d="m6 6 12 12M18 6 6 18"/></>,
  document: <><path d="M6 2.5h8l4 4v15H6z"/><path d="M14 2.5v5h5M9 12h6M9 16h6"/></>,
  external: <><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6H5V6h6"/></>,
  gear: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.2"/></>,
  library: <><path d="M4 5.5h16v15H4zM8 2.5v6M16 2.5v6M8 12h8M8 16h5"/></>,
  menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
  plus: <><path d="M12 5v14M5 12h14"/></>,
  refresh: <><path d="M20 7v5h-5"/><path d="M19 12a7 7 0 1 0-2 5"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/></>,
  send: <><path d="m3 11 18-8-8 18-2-8zM11 13l10-10"/></>,
  shield: <><path d="M12 2.5 20 6v5.5c0 5-3.4 8.4-8 10-4.6-1.6-8-5-8-10V6z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></>,
  trash: <><path d="M4 7h16M9 7V4h6v3M7 7l1 14h8l1-14M10 11v6M14 11v6"/></>,
  upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
  warning: <><path d="M12 3 2.5 20h19z"/><path d="M12 9v5M12 17.5v.1"/></>,
}

function Icon({ name, size = 20, className = "" }) {
  return (
    <svg
      aria-hidden="true"
      className={`icon ${className}`}
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
    >
      {paths[name]}
    </svg>
  )
}

export default Icon
