// Typed, dependency-free CDP transport for isolated local browser validation.
interface Target {type: string; webSocketDebuggerUrl: string}
interface Pending {resolve(value: unknown): void; reject(error: Error): void}
interface Evaluation<T> {exceptionDetails?: unknown; result: {value: T}}
export class CDP {
  private socket: WebSocket;
  private next = 0;
  private pending = new Map<number, Pending>();
  readonly errors: unknown[] = [];
  static async connect(port: number): Promise<CDP> {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json() as Target[];
    const target = targets.find(value => value.type === 'page');
    if (!target || typeof target.webSocketDebuggerUrl !== 'string') throw new Error('Start an isolated Chrome profile with CDP first.');
    const socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise<void>((resolve, reject) => {
      socket.addEventListener('open', () => resolve(), {once: true});
      socket.addEventListener('error', () => reject(new Error('Could not connect to Chrome.')), {once: true});
    });
    return new CDP(socket);
  }
  private constructor(socket: WebSocket) {
    this.socket = socket;
    socket.addEventListener('message', ({data}: MessageEvent<unknown>) => {
      if (typeof data !== 'string') return;
      const message = JSON.parse(data) as {id?: number; result?: unknown; error?: unknown; method?: string; params?: {exceptionDetails?: unknown}};
      if (message.id) {
        const callback = this.pending.get(message.id); this.pending.delete(message.id);
        if (callback) message.error ? callback.reject(new Error(JSON.stringify(message.error))) : callback.resolve(message.result);
      }
      if (message.method === 'Runtime.exceptionThrown') this.errors.push(message.params?.exceptionDetails);
    });
    socket.addEventListener('close', () => {
      for (const callback of this.pending.values()) callback.reject(new Error('Chrome connection closed.'));
      this.pending.clear();
    });
  }
  call<T = Record<string, unknown>>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const id = ++this.next; this.pending.set(id, {resolve: value => resolve(value as T), reject});
      this.socket.send(JSON.stringify({id, method, params}));
    });
  }
  async evaluate<T = unknown>(expression: string): Promise<T> {
    const result = await this.call<Evaluation<T>>('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  async wait<T = unknown>(expression: string, timeout = 60000): Promise<T> {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      const result = await this.evaluate<T>(expression); if (result) return result;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error(`Timed out waiting for ${expression}`);
  }
  close(): void {this.socket.close();}
}
