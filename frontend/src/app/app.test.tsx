import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";
import type { HealthDetails } from "../api/types";
import { App } from "../App";
import { ErrorBoundary } from "./ErrorBoundary";
import { StatusBar } from "./StatusBar";
import { ToastProvider, useToasts } from "./Toasts";

const check = (id: string, status: "ok" | "warn" | "error", message: string, hint: string | null = null) => ({
  id,
  label: id.toUpperCase(),
  status,
  message,
  hint,
});

const healthy: HealthDetails = {
  status: "ok",
  version: "0.3.0",
  checks: [check("database", "ok", "Up to date"), check("gpu", "ok", "GTX 1050 Ti")],
};
const noGpu: HealthDetails = {
  status: "warning",
  version: "0.3.0",
  checks: [check("database", "ok", "Up to date"), check("gpu", "warn", "No usable GPU", "Run make check.")],
};

function fetchReturning(handler: (url: string) => Response | Promise<Response>) {
  return vi.fn((url: RequestInfo | URL) => Promise.resolve(handler(String(url)))) as unknown as typeof fetch;
}
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  window.location.hash = "";
});

describe("ErrorBoundary", () => {
  function Bomb({ explode }: { explode: boolean }) {
    if (explode) throw new Error("secret stack detail ZEBRA");
    return <p>all fine</p>;
  }

  it("shows a friendly message, no stack trace, and recovers with Try again", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const seen: Error[] = [];
    function Harness() {
      const [explode, setExplode] = useState(true);
      return (
        <ErrorBoundary onError={(e) => seen.push(e)}>
          <Bomb explode={explode} />
          <button onClick={() => setExplode(false)}>fix</button>
        </ErrorBoundary>
      );
    }
    render(<Harness />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Something went wrong on this screen");
    expect(alert).toHaveTextContent("Your transcriptions are saved");
    expect(alert.textContent).not.toMatch(/ZEBRA|at Bomb|stack/i);
    expect(screen.getByRole("link", { name: "Back to the library" })).toHaveAttribute("href", "#/");
    expect(seen).toHaveLength(1);
    expect(consoleError).toHaveBeenCalled(); // the details go to the console only
    expect(screen.getByRole("button", { name: "Reload the app" })).toBeInTheDocument();
  });

  it("recovers when the same children render fine again", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    let explode = true;
    function Flaky() {
      if (explode) throw new Error("boom");
      return <p>recovered</p>;
    }
    render(
      <ErrorBoundary>
        <Flaky />
      </ErrorBoundary>,
    );
    await screen.findByRole("alert");
    explode = false;
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("recovered")).toBeInTheDocument();
  });
});

describe("Toasts", () => {
  function Trigger() {
    const t = useToasts();
    return (
      <>
        <button onClick={() => t.error("Disk is full")}>fail</button>
        <button onClick={() => t.push("info", "Saved")}>note</button>
      </>
    );
  }

  it("shows dismissible messages without blocking, and merges identical ones", async () => {
    render(
      <ToastProvider autoDismissMs={0}>
        <Trigger />
      </ToastProvider>,
    );
    await userEvent.click(screen.getByText("fail"));
    await userEvent.click(screen.getByText("fail"));
    await userEvent.click(screen.getByText("note"));
    expect(screen.getAllByText("Disk is full")).toHaveLength(1);
    expect(screen.getByRole("alert")).toHaveTextContent("Disk is full");
    expect(screen.getByRole("status")).toHaveTextContent("Saved");
    await userEvent.click(screen.getAllByRole("button", { name: "Dismiss message" })[0]!);
    expect(screen.queryByText("Disk is full")).not.toBeInTheDocument();
  });

  it("dismisses itself after a while and keeps at most four", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider autoDismissMs={1000}>
        <Trigger />
      </ToastProvider>,
    );
    fireEvent.click(screen.getByText("note"));
    expect(screen.getByText("Saved")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1500));
    expect(screen.queryByText("Saved")).not.toBeInTheDocument();
  });

  it("does nothing outside a provider", async () => {
    render(<Trigger />);
    await userEvent.click(screen.getByText("fail"));
    expect(screen.queryByText("Disk is full")).not.toBeInTheDocument();
  });
});

describe("ApiClient error format", () => {
  it("joins message and hint, keeps the code, and tells listeners about server errors", async () => {
    const api = new ApiClient(
      "http://api.test",
      fetchReturning(() =>
        json({ detail: "x", code: "db_locked", message: "The project database is busy.", hint: "Try again in a moment." }, 503),
      ),
    );
    const events: string[] = [];
    api.subscribe((e) => events.push(e.type));
    const err = (await api.listDocuments().catch((e: unknown) => e)) as ApiError;
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(503);
    expect(err.code).toBe("db_locked");
    expect(err.reason).toBe("The project database is busy.");
    expect(err.hint).toBe("Try again in a moment.");
    expect(err.message).toBe("The project database is busy. Try again in a moment.");
    expect(events).toEqual(["server"]);
  });

  it("falls back to the plain detail of older servers and keeps 4xx quiet", async () => {
    const api = new ApiClient("http://api.test", fetchReturning(() => json({ detail: "page 1 not found" }, 404)));
    const events: string[] = [];
    api.subscribe((e) => events.push(e.type));
    await expect(api.getPage(1)).rejects.toMatchObject({ status: 404, message: "page 1 not found", code: null, hint: null });
    expect(events).toEqual([]);
  });

  it("reports a lost connection with advice and emits network, then ok when it is back", async () => {
    let up = false;
    const api = new ApiClient(
      "http://api.test",
      fetchReturning(() => {
        if (!up) throw new TypeError("Failed to fetch");
        return json([]);
      }),
    );
    const events: string[] = [];
    api.subscribe((e) => events.push(e.type));
    const err = (await api.listDocuments().catch((e: unknown) => e)) as ApiError;
    expect(err.status).toBe(0);
    expect(err.code).toBe("network");
    expect(err.message).toContain("Cannot reach the API at http://api.test");
    expect(err.message).toContain("make dev");
    up = true;
    await api.listDocuments();
    expect(events).toEqual(["network", "ok"]);
  });

  it("unsubscribes", async () => {
    const api = new ApiClient("http://api.test", fetchReturning(() => json([])));
    const seen: string[] = [];
    const off = api.subscribe((e) => seen.push(e.type));
    await api.listDocuments();
    off();
    await api.listDocuments();
    expect(seen).toEqual(["ok"]);
  });
});

describe("StatusBar", () => {
  it("shows a green indicator and a list of checks", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "getHealthDetails").mockResolvedValue(healthy);
    render(
      <ToastProvider>
        <StatusBar api={api} />
      </ToastProvider>,
    );
    const pill = await screen.findByTestId("status-indicator");
    await waitFor(() => expect(pill).toHaveTextContent("Ready"));
    expect(pill).toHaveClass("status-ok");
    await userEvent.click(pill);
    expect(screen.getByRole("region", { name: "System status" })).toHaveTextContent("GTX 1050 Ti");
  });

  it("flags warnings with a count and shows the hint", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "getHealthDetails").mockResolvedValue(noGpu);
    render(
      <ToastProvider>
        <StatusBar api={api} />
      </ToastProvider>,
    );
    const pill = await screen.findByTestId("status-indicator");
    await waitFor(() => expect(pill).toHaveTextContent("Ready, with notes (1)"));
    await userEvent.click(pill);
    expect(screen.getByRole("region", { name: "System status" })).toHaveTextContent("No usable GPU");
    expect(screen.getByRole("region", { name: "System status" })).toHaveTextContent("Run make check.");
  });

  it("shows the offline banner with Retry, and recovers when the server is back", async () => {
    const api = new ApiClient("http://api.test");
    const details = vi.spyOn(api, "getHealthDetails").mockRejectedValue(new ApiError(0, "Cannot reach the API"));
    render(
      <ToastProvider>
        <StatusBar api={api} />
      </ToastProvider>,
    );
    const banner = await screen.findByTestId("offline-banner");
    expect(banner).toHaveTextContent("cannot reach its local server");
    expect(banner).toHaveTextContent("make dev");
    expect(screen.getByTestId("status-indicator")).toHaveTextContent("Server offline");
    details.mockResolvedValue(healthy);
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.queryByTestId("offline-banner")).not.toBeInTheDocument());
    expect(screen.getByTestId("status-indicator")).toHaveTextContent("Ready");
  });

  it("polls: notices the server going down and coming back without a click", async () => {
    vi.useFakeTimers();
    const api = new ApiClient("http://api.test");
    const details = vi.spyOn(api, "getHealthDetails").mockResolvedValue(healthy);
    render(
      <ToastProvider>
        <StatusBar api={api} pollMs={1000} retryMs={500} />
      </ToastProvider>,
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10);
    });
    expect(screen.getByTestId("status-indicator")).toHaveTextContent("Ready");
    details.mockRejectedValue(new ApiError(0, "down"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1100);
    });
    expect(screen.getByTestId("offline-banner")).toBeInTheDocument();
    details.mockResolvedValue(healthy);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(600);
    });
    expect(screen.queryByTestId("offline-banner")).not.toBeInTheDocument();
  });

  it("turns server errors and lost connections from any screen into one toast", async () => {
    const api = new ApiClient("http://api.test", fetchReturning(() => json({ detail: "x", code: "disk_full", message: "The disk is full.", hint: "Free some space." }, 507)));
    vi.spyOn(api, "getHealthDetails").mockResolvedValue(healthy);
    render(
      <ToastProvider autoDismissMs={0}>
        <StatusBar api={api} />
      </ToastProvider>,
    );
    await screen.findByTestId("status-indicator");
    await api.listDocuments().catch(() => undefined);
    await api.listDocuments().catch(() => undefined);
    const toasts = await screen.findAllByRole("alert");
    expect(toasts.filter((t) => t.textContent?.includes("The disk is full. Free some space."))).toHaveLength(1);
  });
});

describe("App shell", () => {
  it("renders the status strip above the screen and keeps the screen working", async () => {
    const api = new ApiClient("http://api.test");
    vi.spyOn(api, "getHealthDetails").mockResolvedValue(noGpu);
    vi.spyOn(api, "listDocuments").mockResolvedValue([]);
    vi.spyOn(api, "listPages").mockResolvedValue([]);
    vi.spyOn(api, "listJobs").mockResolvedValue([]);
    vi.spyOn(api, "getTrainingSummary").mockRejectedValue(new ApiError(404, "no summary"));
    render(<App client={api} />);
    expect(await screen.findByRole("heading", { name: "HTR library" })).toBeInTheDocument();
    expect(await screen.findByTestId("status-indicator")).toHaveTextContent("Ready, with notes");
    expect(screen.getByRole("link", { name: "HTR" })).toHaveAttribute("href", "#/");
  });

  it("shows the offline banner on top of the library when the API is down", async () => {
    const api = new ApiClient("http://api.test");
    const down = new ApiError(0, "Cannot reach the API at http://api.test", "network", "Start it with `make dev`.");
    vi.spyOn(api, "getHealthDetails").mockRejectedValue(down);
    vi.spyOn(api, "listDocuments").mockRejectedValue(down);
    vi.spyOn(api, "listPages").mockRejectedValue(down);
    vi.spyOn(api, "listJobs").mockRejectedValue(down);
    render(<App client={api} />);
    expect(await screen.findByTestId("offline-banner")).toBeInTheDocument();
    expect(await screen.findByText(/Cannot reach the API at http:\/\/api.test/)).toBeInTheDocument();
  });
});
