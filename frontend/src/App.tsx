import { Abbreviations } from "./abbreviations/Abbreviations";
import { ApiClient } from "./api/client";
import { PageEditor } from "./editor/PageEditor";
import { ExportPage } from "./export/ExportPage";
import { Library } from "./library/Library";
import { Review } from "./review/Review";
import { TextView } from "./text/TextView";
import { Training } from "./training/Training";
import { useRoute } from "./router";

const api = new ApiClient();

export function App({ client = api }: { client?: ApiClient }) {
  const route = useRoute();
  if (route.name === "page") return <PageEditor key={route.id} api={client} pageId={route.id} lineId={route.line} />;
  if (route.name === "text") return <TextView key={route.id} api={client} pageId={route.id} />;
  if (route.name === "training") return <Training api={client} />;
  if (route.name === "review") return <Review api={client} />;
  if (route.name === "abbreviations") return <Abbreviations api={client} />;
  if (route.name === "export") return <ExportPage api={client} />;
  return <Library api={client} />;
}
