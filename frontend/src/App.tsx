import { ApiClient } from "./api/client";
import { PageEditor } from "./editor/PageEditor";
import { Library } from "./library/Library";
import { Training } from "./training/Training";
import { useRoute } from "./router";

const api = new ApiClient();

export function App({ client = api }: { client?: ApiClient }) {
  const route = useRoute();
  if (route.name === "page") return <PageEditor key={route.id} api={client} pageId={route.id} />;
  if (route.name === "training") return <Training api={client} />;
  return <Library api={client} />;
}
