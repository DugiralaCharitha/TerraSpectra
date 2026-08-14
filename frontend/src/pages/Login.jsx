import { useState } from 'react'

function Login({ onLogin })  {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (event) => {
  event.preventDefault()

  // Frontend-only navigation for now.
  // Real credential verification will be connected
  // to the backend later.
  onLogin()
}

  return (
    <main className="login-page">
      <div className="login-card">

        <div className="login-header">
          <h1>TerraSpectra</h1>
          <p>Geospatial Crop Intelligence Platform</p>
        </div>

        <div className="login-title">
          <h2>Welcome Back</h2>
          <p>Sign in to access your farm dashboard.</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">

          <div className="form-group">
            <label htmlFor="email">Email Address</label>

            <input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>

            <input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </div>

          <div className="login-options">
            <label className="remember-me">
              <input type="checkbox" />
              <span>Remember me</span>
            </label>

            <button
              type="button"
              className="forgot-password"
              onClick={() => alert('Password recovery will be connected to the backend later.')}
            >
              Forgot password?
            </button>
          </div>

          <button type="submit" className="login-button">
            Sign In
          </button>

        </form>

        <p className="login-footer">
          Secure access to your TerraSpectra farm data
        </p>

      </div>
    </main>
  )
}

export default Login