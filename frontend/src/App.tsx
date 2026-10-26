import { ApiClient } from "./api/client";
import { PageEditor } from "./editor/PageEditor";
import { Library } from "./library/Library";
import { useRoute } from "./router";

const api = new ApiClient();

export function App({ client = api }: { client?: ApiClient }) {
  const route = useRoute();
  return route.name === "page" ? <PageEditor key={route.id} api={client} pageId={route.id} /> : <Library api={client} />;
}
