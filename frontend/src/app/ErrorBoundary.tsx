import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Called with the error (for tests); the user never sees the stack. */
  onError?: (error: Error) => void;
}

/**
 * Catches a crash while rendering a screen and shows a friendly message instead of a blank page. The technical
 * details go to the browser console only: users never see a stack trace. Unsaved typing in the page editor is
 * saved as it goes, so reloading is safe.
 */
export class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("HTR screen crashed:", error, info.componentStack);
    this.props.onError?.(error);
  }

  render(): ReactNode {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="crash" role="alert">
        <h1>Something went wrong on this screen</h1>
        <p>
          The app hit an unexpected problem while drawing this page. Your transcriptions are saved on this computer as
          you go, so nothing you saved is lost.
        </p>
        <p className="actions">
          <button onClick={() => this.setState({ failed: false })}>Try again</button>
          <button onClick={() => window.location.reload()}>Reload the app</button>
          <a className="button" href="#/">
            Back to the library
          </a>
        </p>
        <p className="muted small">Technical details are in the browser console and in data/logs/htr.log.</p>
      </main>
    );
  }
}
