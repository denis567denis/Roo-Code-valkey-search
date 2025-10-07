import { createClient, RedisClientType } from "redis"
import { createHash } from "crypto"
import * as path from "path"
import * as vscode from "vscode"
import { IVectorStore, VectorStoreSearchResult, Payload } from "../interfaces"
import { DEFAULT_MAX_SEARCH_RESULTS, DEFAULT_SEARCH_MIN_SCORE } from "../constants"
import { t } from "../../../i18n"

export class ValkeySearchVectorStore implements IVectorStore {
	private readonly vectorSize: number
	private readonly DISTANCE_METRIC = "COSINE"
	private client: RedisClientType | null = null
	private isInitializing = false
	private readonly indexName: string
	private readonly valkeyHostname: string
	private readonly valkeyPort: number
	private readonly valkeyUsername?: string
	private readonly valkeyPassword?: string
	private readonly useSsl: boolean
	private readonly workspacePath: string
	private static outputChannel: vscode.OutputChannel | null = null

	static getOutputChannel(): vscode.OutputChannel {
		if (!ValkeySearchVectorStore.outputChannel) {
			ValkeySearchVectorStore.outputChannel = vscode.window.createOutputChannel("ValkeySearch")
		}

		return ValkeySearchVectorStore.outputChannel
	}

	// Метод для логирования в Output Channel
	static log(message: string, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO'): void {
		const timestamp = new Date().toISOString()
		const logMessage = `[${timestamp}] [${level}] ${message}`

		// Выводим в наш специальный Output Channel
		ValkeySearchVectorStore.getOutputChannel().appendLine(logMessage)

		// Также выводим в консоль для совместимости
		console.log(`[ValkeySearch] ${message}`)
	}

	constructor(
		workspacePath: string,
		hostname: string,
		port: number,
		vectorSize: number,
		username?: string,
		password?: string,
		useSsl: boolean = false,
	) {
		this.workspacePath = workspacePath
		this.valkeyHostname = hostname
		this.valkeyPort = port
		this.valkeyUsername = username
		this.valkeyPassword = password
		this.vectorSize = vectorSize
		this.useSsl = useSsl || false

		const hash = createHash("sha256").update(workspacePath).digest("hex")
		this.indexName = `ws-${hash.substring(0, 16)}`
		this.initializeClient()
	}

	private normalizeVector(vector: number[]): number[] {
		const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));

		if (norm === 0) {
			return vector;
		}

		return vector.map(val => val / norm);
	}

	private async initializeClient(): Promise<void> {
		if (this.isInitializing) {
			console.log("[ValkeySearch] Connection already initializing")
			return
		}

		this.isInitializing = true

		try {
			const url = this.useSsl
				? `rediss://${this.valkeyUsername}:${this.valkeyPassword}@${this.valkeyHostname}:${this.valkeyPort}`
				: `redis://${this.valkeyUsername}:${this.valkeyPassword}@${this.valkeyHostname}:${this.valkeyPort}`

			this.client = createClient({
				url
			})

			this.client.on("error", (error: Error) => {
				ValkeySearchVectorStore.log("[ValkeySearch] Connection error: " + error.message)
				this.isInitializing = false
			})

			this.client.on("ready", () => {
				this.isInitializing = false
				ValkeySearchVectorStore.log("[ValkeySearch] Connection established")
			})

			this.client.on("end", () => {
				ValkeySearchVectorStore.log("[ValkeySearch] Connection closed")
				this.destroy()
			})

			await this.client.connect()
		} catch (error) {
			this.isInitializing = false
			ValkeySearchVectorStore.log("[ValkeySearch] Connection failed:", error.message)
			if (error instanceof Error) {
				throw new Error(
					t("embeddings:vectorStore.valkeyConnectionFailed", {
						valkeyUrl: `${this.valkeyHostname}:${this.valkeyPort}`,
						errorMessage: error.message,
					}),
				)
			}
			throw error
		}
	}

	private async ensureConnected() {
		if (!this.client || !this.client.isReady) {
			await this.initializeClient()
		}
	}

	async initialize(): Promise<boolean> {
		await this.ensureConnected()

		try {
			const infoArray = await this.client?.ft.info(this.indexName)
			ValkeySearchVectorStore.log("FT INFO result: " + JSON.stringify(infoArray))
			const dimension = await this.getIndexDimension(this.indexName)

			if (Array.isArray(infoArray) && dimension === this.vectorSize) {
				return false
			} else {
				await this.deleteCollection()
			}
		} catch (error) {
			// Index does not exist, continue creation
		}

		try {
			await this._createIndex()
			await this.saveIndexDimension(this.indexName, this.vectorSize)
			return true
		} catch (error: any) {
			throw new Error(error.message)
		}
	}

	private async _createIndex(): Promise<void> {
		await this.client?.ft.create(this.indexName,
			{
				'$.embedding': {
					type: 'VECTOR',
					ALGORITHM: 'HNSW',
					TYPE: 'FLOAT32',
					DISTANCE_METRIC: 'COSINE',
					DIM: this.vectorSize,
					M: 64,
					EF_CONSTRUCTION: 512,
					AS: 'embedding'
				},
				'$.pathSegments': {
					type: 'TAG',
					SEPARATOR: '|',
					CASESENSITIVE: true,
					AS: 'pathSegments'
				},
				'$.filePath': {
					type: 'TAG',
					CASESENSITIVE: true,
					AS: 'filePath'
				}
			},
			{
				ON: 'JSON',

			})
		await this.saveIndexDimension(this.indexName, this.vectorSize)
	}

	buildPathSegments(filePath?: string): string[] {
		if (!filePath) return []

		const normalizedPath = path.posix.normalize(filePath.replace(/\\/g, "/"))

		if (normalizedPath === "/" || normalizedPath === "." || normalizedPath === "./") return ["/"]

		const parts = normalizedPath.split("/").filter(Boolean)
		const prefixes: string[] = []
		for (let i = 0; i < parts.length; i++) {
			const pref = "/" + parts.slice(0, i + 1).join("/")
			prefixes.push(pref)
		}
		return prefixes
	}

	async upsertPoints(
		points: Array<{
			id: string
			vector: number[]
			payload: Record<string, any>
		}>,
	): Promise<void> {
		await this.ensureConnected()

		if (points.length === 0) return

		const multi = this.client?.multi()
		for (const point of points) {
			const docId = `${this.indexName}:${point.id}`

			const pathSegments = this.buildPathSegments(point.payload?.filePath)

			const args: Record<string, string | number[]> = {
				filePath: point.payload.filePath,
				pathSegments: pathSegments.join("|"),
				codeChunk: point.payload.codeChunk,
				startLine: String(point.payload.startLine),
				endLine: String(point.payload.endLine),
				embedding: point.vector,
			}

			multi?.json.set(docId, '$', args)
		}
		await multi?.exec()
	}

	float32Buffer(arr: number[]) {
		const floatArray = new Float32Array(arr)
		return Buffer.from(floatArray.buffer)
	}

	async search(
		queryVector: number[],
		directoryPrefix?: string,
		minScore?: number,
		maxResults?: number,
	): Promise<VectorStoreSearchResult[]> {
		await this.ensureConnected()

		let directoryFilter: string | undefined = undefined
		if (directoryPrefix) {
			const normalizedPrefix = path.posix.normalize(directoryPrefix.replace(/\\/g, "/"))
			if (normalizedPrefix !== "." && normalizedPrefix !== "./") {
				directoryFilter = normalizedPrefix.replace(/\./g, "\\.").replace(/\//g, "\\/").replace(/-/g, "\\-")
			}
		}

		const vectorBuffer = this.float32Buffer(queryVector)
		const searchLimit = maxResults ?? DEFAULT_MAX_SEARCH_RESULTS

		const q = directoryFilter
			? `@pathSegments:{${directoryFilter}*} => [KNN ${searchLimit} @embedding $B EF_RUNTIME 200 AS score]`
			: `*=>[KNN ${searchLimit} @embedding $B EF_RUNTIME 200 AS score]`

		ValkeySearchVectorStore.log('> Search query: ' + queryVector.length)

		const results = await this.client?.ft.search(this.indexName, q,
			{
				PARAMS: {
					B: vectorBuffer,
				},
				RETURN: ["score", "$.filePath", "$.codeChunk", "$.startLine", "$.endLine"],
				DIALECT: 2,
				LIMIT: {
					from: 0,
					size: searchLimit,
				}
			}
		)


		if (!results?.documents) {
			return []
		}

		ValkeySearchVectorStore.log('> Search results: ' + results.documents.length)

		const parsedResults: VectorStoreSearchResult[] = []

		results?.documents.forEach((doc) => {
			ValkeySearchVectorStore.log('doc ' + JSON.stringify(doc))
			parsedResults.push({
				id: doc.id.replace(`${this.indexName}:`, ""),
				payload: {
					filePath: doc.value['$.filePath'] as string,
					codeChunk: doc.value['$.codeChunk'] as string,
					startLine: parseInt(doc.value['$.startLine'] as string),
					endLine: parseInt(doc.value['$.endLine'] as string),
				},
				score: 1 - parseFloat(String(doc.value.score))
			})
		})

		return parsedResults
			.filter((r) => r.score >= (minScore || DEFAULT_SEARCH_MIN_SCORE))
			.sort((a, b) => b.score - a.score)
	}

	async deletePointsByFilePath(filePath: string): Promise<void> {
		await this.deletePointsByMultipleFilePaths([filePath])
	}

	async deletePointsByMultipleFilePaths(filePaths: string[]): Promise<void> {
		if (filePaths.length === 0) return
		await this.ensureConnected()

		try {
			const collectionExists = await this.collectionExists()
			if (!collectionExists) {
				return
			}
			const workspaceRoot = this.workspacePath

			const normalizedFilePaths = filePaths.map((filePath) => {
				const relativePath = path.isAbsolute(filePath) ? path.relative(workspaceRoot, filePath) : filePath
				const normalizedRelativePath = relativePath.startsWith("/") ? relativePath.slice(1) : relativePath
				return `${path.posix.normalize(normalizedRelativePath.replace(/\\/g, "/")).replace(/\./g, "\\.").replace(/\//g, "\\/").replace(/-/g, "\\-")}`
			})
			const query = `@filePath:{${normalizedFilePaths.join("|")}}`

			const result = await this.client?.ft.search(this.indexName, query, {
				LIMIT: {
					from: 0,
					size: 10000,
				}
			})

			const multi = this.client?.multi()
			if (result?.documents) {
				result?.documents.forEach((doc) => {
					multi?.del(doc.id)
				})
				await multi?.exec()
			}
		} catch (error) {
			console.error("Failed to delete points by file paths:", error)
			throw error
		}
	}

	async deleteCollection(): Promise<void> {
		await this.ensureConnected()
		await this.clearCollection()
		await this.client?.sendCommand(["FT.DROPINDEX", this.indexName])
		await this.removeIndexDimension(this.indexName)
	}

	async clearCollection(): Promise<void> {
		try {
			await this.ensureConnected()
			const result = await this.client?.sendCommand([
				"FT.SEARCH",
				this.indexName,
				"*",
				"NOCONTENT",
				"LIMIT",
				"0",
				"1000000",
			])

			if (Array.isArray(result)) {
				ValkeySearchVectorStore.log('> Clear collection: ' + result.length)
				if (result.length > 1) {
					const multi = this.client?.multi()
					for (let i = 1; i < result.length; i++) {
						const docId = result[i] as string
						multi?.json.del(docId)
					}
					await multi?.exec()
					ValkeySearchVectorStore.log('> Clear collection: done ')
				}
			}
		} catch (error: any) {
			ValkeySearchVectorStore.log("Failed to clear collection: " + error.message, "ERROR")
		}
	}

	async collectionExists(): Promise<boolean> {
		await this.ensureConnected()
		try {
			const result = await this.client?.ft.info(this.indexName)
			ValkeySearchVectorStore.log("FT INFO result: " + JSON.stringify(result))
			return true
		} catch (error) {
			return false
		}
	}

	/**
	 * Save index dimension to Redis/Valkey using JSON
	 * @param indexName - index name
	 * @param dimension - dimension
	 */
	async saveIndexDimension(indexName: string, dimension: number): Promise<void> {
		await this.ensureConnected()
		const key = `index:meta:${indexName}`
		const metadata = {
			dimension: dimension,
			createdAt: new Date().toISOString(),
			distanceMetric: this.DISTANCE_METRIC,
		}
		await this.client?.json.set(key, "$", metadata)
	}

	/**
	 * Get index dimension from Redis/Valkey
	 * @param indexName - index name
	 * @returns dimension or null if not found
	 */
	async getIndexDimension(indexName: string): Promise<number | null> {
		await this.ensureConnected()
		try {
			const key = `index:meta:${indexName}`

			const result = await this.client?.json.get(key, {
				path: "$.dimension",
			})

			if (result && typeof result === "string") {

				return Number(result)
			}

			if (result && typeof result === "number") {

				return result
			}

			return null
		} catch (error) {
			return null
		}
	}

	async removeIndexDimension(indexName: string): Promise<void> {
		await this.ensureConnected()
		const key = `index:meta:${indexName}`
		await this.client?.json.del(key)
	}

	async destroy() {
		if (this.client && this.client.disconnect) {
			this.client.disconnect()
		}
	}

	// Статический метод для очистки Output Channel
	static disposeOutputChannel(): void {
		if (ValkeySearchVectorStore.outputChannel) {
			ValkeySearchVectorStore.outputChannel.dispose()
			ValkeySearchVectorStore.outputChannel = null
		}
	}
}
