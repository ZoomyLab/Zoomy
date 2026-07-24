import { VSCodeThemeWrapper } from 'baukasten-ui-web-wrapper'
import {
  Button, Heading, Input, Badge, Alert, Divider, Code, Checkbox, Icon,
} from 'baukasten-ui'

const V = (name, fb) => `var(--vscode-${name}${fb ? ', ' + fb : ''})`
const models = [
  { id: 'swe',   name: 'Shallow Water',    desc: 'SME(level=0)' },
  { id: 'sme',   name: 'Shallow Moments',  desc: 'SME(level=2)' },
  { id: 'mlsme', name: 'Multilayer SME',   desc: 'MLSME(2 layers)' },
  { id: 'vam',   name: 'Vertically-Averaged Moments', desc: 'VAM(level=1)' },
]

function Shell() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh',
                  background: V('editor-background', '#1e1e1e'), color: V('foreground', '#ccc'),
                  fontFamily: V('font-family', 'system-ui, sans-serif'), fontSize: 13 }}>
      {/* title bar */}
      <div style={{ height: 36, display: 'flex', alignItems: 'center', gap: 10, padding: '0 12px',
                    background: V('titleBar-activeBackground', '#323233'),
                    borderBottom: `1px solid ${V('panel-border', '#2b2b2b')}` }}>
        <strong style={{ letterSpacing: .3 }}>Zoomy</strong>
        <Badge variant="default" size="sm">Baukasten preview</Badge>
        <span style={{ marginLeft: 'auto', opacity: .65, fontSize: 12 }}>
          backend-less · runs entirely in the browser
        </span>
      </div>

      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* activity bar */}
        <div style={{ width: 48, display: 'flex', flexDirection: 'column', alignItems: 'center',
                      gap: 16, padding: '10px 0', background: V('activityBar-background', '#333'),
                      borderRight: `1px solid ${V('panel-border', '#2b2b2b')}` }}>
          <Icon name="home" /><Icon name="settings-gear" />
        </div>

        {/* sidebar: model list */}
        <div style={{ width: 250, overflow: 'auto', padding: 12,
                      background: V('sideBar-background', '#252526'),
                      borderRight: `1px solid ${V('panel-border', '#2b2b2b')}` }}>
          <Heading level={5}>Models</Heading>
          <div style={{ marginTop: 8 }}>
            {models.map((m) => (
              <div key={m.id} style={{ padding: '8px 10px', borderRadius: 4, marginBottom: 4, cursor: 'pointer',
                background: m.id === 'sme' ? V('list-activeSelectionBackground', '#094771') : 'transparent' }}>
                <div style={{ fontWeight: 600 }}>{m.name}</div>
                <div style={{ fontSize: 12, opacity: .65 }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* editor / config */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          <Heading level={3}>Shallow Moments — SME(level = 2)</Heading>
          <p style={{ opacity: .8, maxWidth: 620, lineHeight: 1.5 }}>
            Configure and run entirely in your browser. This shell is built with TypeFox{' '}
            <Code>baukasten-ui</Code> — the same component code drops into VS Code or Theia
            unchanged (only the stylesheet import differs).
          </p>
          <Divider />
          <div style={{ display: 'grid', gridTemplateColumns: '180px 220px', gap: 12,
                        alignItems: 'center', marginTop: 18, maxWidth: 420 }}>
            <span>Bulk viscosity ν</span>       <Input defaultValue="0.1" />
            <span>Bed slip λ<sub>s</sub></span> <Input defaultValue="0.5" />
            <span>Moment level</span>           <Input defaultValue="2" />
          </div>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, marginTop: 16 }}>
            <Checkbox defaultChecked /> Plot the vertical velocity profile
          </label>
          <div style={{ marginTop: 22, display: 'flex', gap: 10 }}>
            <Button variant="primary"><Icon name="play" /> Run in browser</Button>
            <Button variant="secondary">Open in Jupyter</Button>
          </div>
          <div style={{ marginTop: 26, maxWidth: 620 }}>
            <Alert variant="info" title="Backend-less preview">
              No server and no editor host. The VS Code look comes from
              baukasten-ui-web-wrapper — try the theme switcher (top-right).
            </Alert>
          </div>
        </div>
      </div>

      {/* status bar */}
      <div style={{ height: 24, display: 'flex', alignItems: 'center', gap: 14, padding: '0 12px',
                    fontSize: 12, background: V('statusBar-background', '#007acc'),
                    color: V('statusBar-foreground', '#fff') }}>
        <span>NumPy (Pyodide)</span><span>mass ✓</span><span style={{ marginLeft: 'auto' }}>Zoomy · Baukasten POC</span>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <VSCodeThemeWrapper showThemeSelector>
      <Shell />
    </VSCodeThemeWrapper>
  )
}
