export default function StubRoom(props: { name: string }) {
  return (
    <div>
      <h3>{props.name}</h3>
      <p style={{ opacity: 0.6 }}>Coming soon. Planned panels for this room:</p>
      <ul style={{ opacity: 0.6 }}>
        <li>Status overview</li><li>Key actions</li><li>Linked sources</li>
      </ul>
    </div>
  );
}
