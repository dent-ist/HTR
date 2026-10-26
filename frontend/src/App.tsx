import { Abbreviations } from "./abbreviations/Abbreviations";
import { ApiClient } from "./api/client";
import { ErrorBoundary } from "./app/ErrorBoundary";
import { StatusBar } from "./app/StatusBar";
import { ToastProvider } from "./app/Toasts";
import { PageEditor } from "./editor/PageEditor";
import { ExportPage } from "./export/ExportPage";
import { Library } from "./library/Library";
import { Review } from "./review/Review";
import { TextView } from "./text/TextView";
import { Training } from "./training/Training";
import { useRoute, type Route } from "./router";

const api = new ApiClient();

function Screen({ route, client }: { route: Route; client: ApiClient }) {
  if (route.name === "page") return <PageEditor key={route.id} api={client} pageId={route.id} lineId={route.line} />;
  if (route.name === "text") return <TextView key={route.id} api={client} pageId={route.id} />;
  if (route.name === "training") return <Training api={client} />;
  if (route.name === "review") return <Review api={client} />;
  if (route.name === "abbreviations") return <Abbreviations api={client} />;
  if (route.name === "export") return <ExportPage api={client} />;
  return <Library api={client} />;
}

/**
 * The whole app: a status strip (health indicator, offline banner, error toasts) above the current screen. A crash
 * inside a screen is caught by the error boundary, which resets when the user navigates elsewhere.
 */
export function App({ client = api }: { client?: ApiClient }) {
  const route = useRoute();
  return (
    <ToastProvider>
      <StatusBar api={client} />
      <div className="app-body">
        <ErrorBoundary key={JSON.stringify(route)}>
          <Screen route={route} client={client} />
        </ErrorBoundary>
      </div>
    </ToastProvider>
  );
}
