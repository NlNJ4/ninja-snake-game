'use client';

import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

type Direction = "up" | "down" | "left" | "right";
type GameStatus = "idle" | "running" | "paused" | "over";
type FocusMode = "manual" | "rl";
type LabMode = "idle" | "training" | "watching";
type CrashCause = "self" | "timeout" | "wall";
type AgentAction = "forward" | "left" | "right";
type Point = {
  x: number;
  y: number;
};
type ManualGameState = {
  direction: Direction;
  food: Point;
  queuedDirection: Direction | null;
  score: number;
  snake: Point[];
  status: GameStatus;
};
type SimulationState = {
  cause: CrashCause | null;
  direction: Direction;
  food: Point;
  score: number;
  snake: Point[];
  status: "running" | "over";
  stepsAlive: number;
  stepsSinceFood: number;
};
type Observation = {
  dangers: {
    forward: boolean;
    left: boolean;
    right: boolean;
  };
  foodRelation: {
    x: number;
    y: number;
  };
  heading: Direction;
  key: string;
};
type EpisodeSummary = {
  episode: number;
  reward: number;
  score: number;
};
type LabController = {
  bestScore: number;
  board: SimulationState;
  episodeReward: number;
  episodes: number;
  epsilon: number;
  gamma: number;
  history: EpisodeSummary[];
  lastAction: AgentAction | null;
  lastObservation: Observation;
  lastOutcome: string;
  learningRate: number;
  mode: LabMode;
  qTable: Record<string, [number, number, number]>;
};
type LabSnapshot = {
  averageReward: number;
  averageScore: number;
  bestScore: number;
  board: SimulationState;
  currentObservation: Observation;
  episodes: number;
  epsilon: number;
  history: EpisodeSummary[];
  knownStates: number;
  lastAction: AgentAction | null;
  lastOutcome: string;
  mode: LabMode;
  qValues: [number, number, number];
};
type StepResult = {
  event: "crash" | "food" | "move" | "timeout";
  next: SimulationState;
  reward: number;
};

const BOARD_SIZE = 15;
const INITIAL_LENGTH = 4;
const STORAGE_KEY = "shadow-coil-best-score";
const BEST_SCORE_EVENT = "shadow-coil-best-score-change";
const LAB_HISTORY_LIMIT = 32;
const LAB_BATCH_SIZE = 32;
const LAB_WATCH_INTERVAL = 115;
const LAB_MIN_EPSILON = 0.05;
const LAB_EPSILON_DECAY = 0.995;
const LAB_TIMEOUT_BASE = 80;
const LAB_TIMEOUT_PER_FOOD = 20;
const DEFAULT_DIRECTION: Direction = "right";
const DIRECTION_ORDER: Direction[] = ["up", "right", "down", "left"];
const AGENT_ACTIONS: AgentAction[] = ["forward", "left", "right"];
const DIRECTION_OFFSETS: Record<Direction, Point> = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

function getOppositeDirection(direction: Direction): Direction {
  switch (direction) {
    case "up":
      return "down";
    case "down":
      return "up";
    case "left":
      return "right";
    case "right":
      return "left";
  }
}

function turnDirection(direction: Direction, turn: -1 | 0 | 1): Direction {
  const index = DIRECTION_ORDER.indexOf(direction);

  return DIRECTION_ORDER[(index + turn + DIRECTION_ORDER.length) % DIRECTION_ORDER.length];
}

function directionFromAction(
  direction: Direction,
  action: AgentAction
): Direction {
  switch (action) {
    case "forward":
      return direction;
    case "left":
      return turnDirection(direction, -1);
    case "right":
      return turnDirection(direction, 1);
  }
}

function randomDirection(): Direction {
  return DIRECTION_ORDER[Math.floor(Math.random() * DIRECTION_ORDER.length)];
}

function buildSnake(direction: Direction): Point[] {
  const center = Math.floor(BOARD_SIZE / 2);
  const tailOffset = DIRECTION_OFFSETS[getOppositeDirection(direction)];

  return Array.from({ length: INITIAL_LENGTH }, (_, index) => ({
    x: center + tailOffset.x * index,
    y: center + tailOffset.y * index,
  }));
}

function createPreviewGame(): ManualGameState {
  const snake = buildSnake(DEFAULT_DIRECTION);

  return {
    direction: DEFAULT_DIRECTION,
    food: { x: BOARD_SIZE - 3, y: Math.floor(BOARD_SIZE / 2) },
    queuedDirection: null,
    score: 0,
    snake,
    status: "idle",
  };
}

function spawnFood(snake: Point[]): Point {
  const occupied = new Set(snake.map((segment) => `${segment.x}:${segment.y}`));
  const available: Point[] = [];

  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      if (!occupied.has(`${x}:${y}`)) {
        available.push({ x, y });
      }
    }
  }

  return available[Math.floor(Math.random() * available.length)] ?? {
    x: 0,
    y: 0,
  };
}

function createRunningGame(direction: Direction): ManualGameState {
  const snake = buildSnake(direction);

  return {
    direction,
    food: spawnFood(snake),
    queuedDirection: null,
    score: 0,
    snake,
    status: "running",
  };
}

function createSimulation(direction: Direction = randomDirection()): SimulationState {
  const snake = buildSnake(direction);

  return {
    cause: null,
    direction,
    food: spawnFood(snake),
    score: 0,
    snake,
    status: "running",
    stepsAlive: 0,
    stepsSinceFood: 0,
  };
}

function getStepInterval(score: number): number {
  return Math.max(80, 190 - score * 8);
}

function formatStat(value: number): string {
  return value.toString().padStart(2, "0");
}

function formatSignedMetric(value: number): string {
  const prefix = value > 0 ? "+" : "";

  return `${prefix}${value.toFixed(2)}`;
}

function statusLabel(status: GameStatus): string {
  switch (status) {
    case "idle":
      return "Standby";
    case "running":
      return "Live Run";
    case "paused":
      return "Paused";
    case "over":
      return "Wiped";
  }
}

function labStatusLabel(mode: LabMode): string {
  switch (mode) {
    case "idle":
      return "RL Ready";
    case "training":
      return "Learning";
    case "watching":
      return "Policy Demo";
  }
}

function overlayCopy(status: Exclude<GameStatus, "running">): {
  action: string;
  body: string;
  title: string;
} {
  switch (status) {
    case "idle":
      return {
        action: "Start run",
        body: "Use arrow keys, WASD, or the control pad to launch the coil.",
        title: "Ready stance",
      };
    case "paused":
      return {
        action: "Resume run",
        body: "The board is frozen. Resume when you want another pass.",
        title: "Flow held",
      };
    case "over":
      return {
        action: "Restart run",
        body: "You clipped a wall or crossed your own trail.",
        title: "Mission failed",
      };
  }
}

function labOverlayCopy(mode: LabMode): {
  action: string;
  body: string;
  title: string;
} | null {
  if (mode !== "idle") {
    return null;
  }

  return {
    action: "Start training",
    body: "The agent sees nearby danger plus the food direction, then updates its Q-values after every move.",
    title: "Q-learning lab",
  };
}

function isOppositeDirection(current: Direction, next: Direction): boolean {
  return getOppositeDirection(current) === next;
}

function readBestScoreSnapshot(): number {
  if (typeof window === "undefined") {
    return 0;
  }

  const storedScore = Number(window.localStorage.getItem(STORAGE_KEY));

  return Number.isFinite(storedScore) && storedScore > 0 ? storedScore : 0;
}

function subscribeBestScore(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }

  const handler = () => {
    onStoreChange();
  };

  window.addEventListener("storage", handler);
  window.addEventListener(BEST_SCORE_EVENT, handler);

  return () => {
    window.removeEventListener("storage", handler);
    window.removeEventListener(BEST_SCORE_EVENT, handler);
  };
}

function writeBestScore(score: number) {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.setItem(STORAGE_KEY, String(score));
  window.dispatchEvent(new Event(BEST_SCORE_EVENT));
}

function manhattanDistance(a: Point, b: Point): number {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

function willCollide(state: SimulationState, direction: Direction): boolean {
  const offset = DIRECTION_OFFSETS[direction];
  const nextHead = {
    x: state.snake[0].x + offset.x,
    y: state.snake[0].y + offset.y,
  };
  const willEat =
    nextHead.x === state.food.x && nextHead.y === state.food.y;
  const bodyToCheck = willEat ? state.snake : state.snake.slice(0, -1);
  const hitWall =
    nextHead.x < 0 ||
    nextHead.x >= BOARD_SIZE ||
    nextHead.y < 0 ||
    nextHead.y >= BOARD_SIZE;
  const hitSelf = bodyToCheck.some(
    (segment) => segment.x === nextHead.x && segment.y === nextHead.y
  );

  return hitWall || hitSelf;
}

function getObservation(state: SimulationState): Observation {
  const head = state.snake[0];
  const foodRelation = {
    x: Math.sign(state.food.x - head.x),
    y: Math.sign(state.food.y - head.y),
  };
  const dangers = {
    forward: willCollide(state, state.direction),
    left: willCollide(state, turnDirection(state.direction, -1)),
    right: willCollide(state, turnDirection(state.direction, 1)),
  };

  return {
    dangers,
    foodRelation,
    heading: state.direction,
    key: `${state.direction}|${foodRelation.x},${foodRelation.y}|${Number(dangers.forward)}${Number(dangers.left)}${Number(dangers.right)}`,
  };
}

function stepSimulation(
  state: SimulationState,
  nextDirection: Direction
): StepResult {
  const offset = DIRECTION_OFFSETS[nextDirection];
  const nextHead = {
    x: state.snake[0].x + offset.x,
    y: state.snake[0].y + offset.y,
  };
  const willEat =
    nextHead.x === state.food.x && nextHead.y === state.food.y;
  const bodyToCheck = willEat ? state.snake : state.snake.slice(0, -1);
  const hitWall =
    nextHead.x < 0 ||
    nextHead.x >= BOARD_SIZE ||
    nextHead.y < 0 ||
    nextHead.y >= BOARD_SIZE;
  const hitSelf = bodyToCheck.some(
    (segment) => segment.x === nextHead.x && segment.y === nextHead.y
  );

  if (hitWall || hitSelf) {
    return {
      event: "crash",
      next: {
        ...state,
        cause: hitWall ? "wall" : "self",
        direction: nextDirection,
        status: "over",
      },
      reward: -18,
    };
  }

  const snake = [nextHead, ...state.snake];

  if (!willEat) {
    snake.pop();
  }

  const score = willEat ? state.score + 1 : state.score;
  const previousDistance = manhattanDistance(state.snake[0], state.food);
  const nextDistance = willEat ? 0 : manhattanDistance(nextHead, state.food);
  const stepsSinceFood = willEat ? 0 : state.stepsSinceFood + 1;
  const timeoutLimit = LAB_TIMEOUT_BASE + state.score * LAB_TIMEOUT_PER_FOOD;
  let reward = -0.18;

  if (willEat) {
    reward += 16;
  } else if (nextDistance < previousDistance) {
    reward += 0.9;
  } else if (nextDistance > previousDistance) {
    reward -= 0.45;
  } else {
    reward -= 0.1;
  }

  const nextState: SimulationState = {
    cause: null,
    direction: nextDirection,
    food: willEat ? spawnFood(snake) : state.food,
    score,
    snake,
    status: "running",
    stepsAlive: state.stepsAlive + 1,
    stepsSinceFood,
  };

  if (stepsSinceFood > timeoutLimit) {
    return {
      event: "timeout",
      next: {
        ...nextState,
        cause: "timeout",
        status: "over",
      },
      reward: reward - 10,
    };
  }

  return {
    event: willEat ? "food" : "move",
    next: nextState,
    reward,
  };
}

function actionIndex(action: AgentAction): 0 | 1 | 2 {
  switch (action) {
    case "forward":
      return 0;
    case "left":
      return 1;
    case "right":
      return 2;
  }
}

function ensureQRow(
  table: Record<string, [number, number, number]>,
  key: string
): [number, number, number] {
  if (!table[key]) {
    table[key] = [0, 0, 0];
  }

  return table[key];
}

function pickGreedyAction(row: [number, number, number]): AgentAction {
  const highest = Math.max(...row);
  const contenders = AGENT_ACTIONS.filter(
    (action) => row[actionIndex(action)] === highest
  );

  return contenders[Math.floor(Math.random() * contenders.length)] ?? "forward";
}

function pickAction(
  table: Record<string, [number, number, number]>,
  observation: Observation,
  epsilon: number
): AgentAction {
  if (Math.random() < epsilon) {
    return AGENT_ACTIONS[Math.floor(Math.random() * AGENT_ACTIONS.length)] ?? "forward";
  }

  return pickGreedyAction(ensureQRow(table, observation.key));
}

function averageHistory(
  history: EpisodeSummary[],
  metric: "reward" | "score"
): number {
  if (history.length === 0) {
    return 0;
  }

  const window = history.slice(-12);
  const total = window.reduce((sum, entry) => sum + entry[metric], 0);

  return total / window.length;
}

function roundMetric(value: number): number {
  return Math.round(value * 100) / 100;
}

function createLabController(): LabController {
  const board = createSimulation(DEFAULT_DIRECTION);
  const observation = getObservation(board);

  return {
    bestScore: 0,
    board,
    episodeReward: 0,
    episodes: 0,
    epsilon: 1,
    gamma: 0.92,
    history: [],
    lastAction: null,
    lastObservation: observation,
    lastOutcome: "Model initialized",
    learningRate: 0.18,
    mode: "idle",
    qTable: {},
  };
}

function snapshotBoard(board: SimulationState): SimulationState {
  return {
    ...board,
    food: { ...board.food },
    snake: board.snake.map((segment) => ({ ...segment })),
  };
}

function snapshotObservation(observation: Observation): Observation {
  return {
    dangers: { ...observation.dangers },
    foodRelation: { ...observation.foodRelation },
    heading: observation.heading,
    key: observation.key,
  };
}

function createLabSnapshot(lab: LabController): LabSnapshot {
  const currentObservation = getObservation(lab.board);
  const qValues = [...ensureQRow(lab.qTable, currentObservation.key)] as [
    number,
    number,
    number,
  ];

  return {
    averageReward: averageHistory(lab.history, "reward"),
    averageScore: averageHistory(lab.history, "score"),
    bestScore: lab.bestScore,
    board: snapshotBoard(lab.board),
    currentObservation: snapshotObservation(currentObservation),
    episodes: lab.episodes,
    epsilon: lab.epsilon,
    history: lab.history.map((entry) => ({ ...entry })),
    knownStates: Object.keys(lab.qTable).length,
    lastAction: lab.lastAction,
    lastOutcome: lab.lastOutcome,
    mode: lab.mode,
    qValues,
  };
}

function finalizeEpisode(lab: LabController) {
  lab.episodes += 1;
  lab.bestScore = Math.max(lab.bestScore, lab.board.score);
  lab.history = [
    ...lab.history.slice(-(LAB_HISTORY_LIMIT - 1)),
    {
      episode: lab.episodes,
      reward: roundMetric(lab.episodeReward),
      score: lab.board.score,
    },
  ];
  lab.epsilon = Math.max(LAB_MIN_EPSILON, lab.epsilon * LAB_EPSILON_DECAY);

  if (lab.board.cause === "timeout") {
    lab.lastOutcome = "Episode reset after stalling";
  } else if (lab.board.cause === "wall") {
    lab.lastOutcome = "Episode ended on a wall hit";
  } else {
    lab.lastOutcome = "Episode ended on self collision";
  }

  lab.board = createSimulation(randomDirection());
  lab.episodeReward = 0;
  lab.lastAction = null;
  lab.lastObservation = getObservation(lab.board);
}

function trainLabStep(lab: LabController) {
  const observation = getObservation(lab.board);
  const action = pickAction(lab.qTable, observation, lab.epsilon);
  const nextDirection = directionFromAction(lab.board.direction, action);
  const result = stepSimulation(lab.board, nextDirection);
  const row = ensureQRow(lab.qTable, observation.key);
  const nextValue =
    result.next.status === "over"
      ? 0
      : Math.max(...ensureQRow(lab.qTable, getObservation(result.next).key));
  const index = actionIndex(action);
  const updated =
    row[index] +
    lab.learningRate *
      (result.reward + lab.gamma * nextValue - row[index]);

  row[index] = updated;
  lab.board = result.next;
  lab.episodeReward += result.reward;
  lab.lastAction = action;
  lab.lastObservation = observation;

  if (result.event === "food") {
    lab.lastOutcome = "Reward spike: food collected";
  } else if (result.event === "timeout") {
    lab.lastOutcome = "Penalty applied: no food for too long";
  } else if (result.event === "crash") {
    lab.lastOutcome = "Penalty applied: collision";
  } else {
    lab.lastOutcome =
      Math.abs(result.reward) < 0.3
        ? "Small step cost"
        : "Shaped reward updated";
  }

  if (result.next.status === "over") {
    finalizeEpisode(lab);
  }
}

function watchLabStep(lab: LabController) {
  if (lab.board.status === "over") {
    lab.board = createSimulation(DEFAULT_DIRECTION);
  }

  const observation = getObservation(lab.board);
  const action = pickGreedyAction(ensureQRow(lab.qTable, observation.key));
  const nextDirection = directionFromAction(lab.board.direction, action);
  const result = stepSimulation(lab.board, nextDirection);

  lab.board = result.next.status === "over" ? createSimulation(randomDirection()) : result.next;
  lab.lastAction = action;
  lab.lastObservation = observation;

  if (result.event === "food") {
    lab.lastOutcome = "Policy took a food reward";
  } else if (result.event === "move") {
    lab.lastOutcome = "Policy selecting highest-value action";
  } else if (result.event === "timeout") {
    lab.lastOutcome = "Demo episode reset after timeout";
  } else {
    lab.lastOutcome = "Demo episode reset after collision";
  }
}

function describeHeading(direction: Direction): string {
  switch (direction) {
    case "up":
      return "North";
    case "down":
      return "South";
    case "left":
      return "West";
    case "right":
      return "East";
  }
}

function describeFoodAxis(
  axis: number,
  negative: string,
  positive: string
): string {
  if (axis < 0) {
    return negative;
  }

  if (axis > 0) {
    return positive;
  }

  return "aligned";
}

function actionLabel(action: AgentAction | null): string {
  if (!action) {
    return "Waiting";
  }

  switch (action) {
    case "forward":
      return "Go forward";
    case "left":
      return "Turn left";
    case "right":
      return "Turn right";
  }
}

function chartRange(values: number[]): { max: number; min: number } {
  if (values.length === 0) {
    return { max: 1, min: 0 };
  }

  return {
    max: Math.max(...values),
    min: Math.min(...values),
  };
}

export default function SnakeGame() {
  const [viewMode, setViewMode] = useState<FocusMode>("manual");
  const [game, setGame] = useState<ManualGameState>(() => createPreviewGame());
  const [labController] = useState<LabController>(() => createLabController());
  const labRef = useRef<LabController>(labController);
  const [lab, setLab] = useState<LabSnapshot>(() =>
    createLabSnapshot(labController)
  );
  const persistedBestScore = useSyncExternalStore(
    subscribeBestScore,
    readBestScoreSnapshot,
    () => 0
  );

  const syncLabView = () => {
    const snapshot = createLabSnapshot(labRef.current);

    startTransition(() => {
      setLab(snapshot);
    });
  };

  const setLabMode = (mode: LabMode) => {
    labRef.current.mode = mode;
    syncLabView();
  };

  const resetLab = () => {
    labRef.current = createLabController();
    syncLabView();
  };

  const startRun = (direction: Direction = DEFAULT_DIRECTION) => {
    setGame(() => createRunningGame(direction));
  };

  const resetBoard = () => {
    setGame(() => createPreviewGame());
  };

  const toggleRun = () => {
    setGame((current) => {
      if (current.status === "running") {
        return { ...current, status: "paused" };
      }

      if (current.status === "paused") {
        return { ...current, status: "running" };
      }

      return createRunningGame(current.direction);
    });
  };

  const steer = (direction: Direction) => {
    setGame((current) => {
      if (current.status === "idle" || current.status === "over") {
        return createRunningGame(direction);
      }

      const activeDirection = current.queuedDirection ?? current.direction;

      if (
        current.status === "paused" ||
        activeDirection === direction ||
        isOppositeDirection(activeDirection, direction)
      ) {
        return current;
      }

      return {
        ...current,
        queuedDirection: direction,
      };
    });
  };

  const stepGame = useEffectEvent(() => {
    setGame((current) => {
      if (current.status !== "running") {
        return current;
      }

      const nextDirection = current.queuedDirection ?? current.direction;
      const offset = DIRECTION_OFFSETS[nextDirection];
      const nextHead = {
        x: current.snake[0].x + offset.x,
        y: current.snake[0].y + offset.y,
      };
      const willEat =
        nextHead.x === current.food.x && nextHead.y === current.food.y;
      const bodyToCheck = willEat ? current.snake : current.snake.slice(0, -1);
      const hitWall =
        nextHead.x < 0 ||
        nextHead.x >= BOARD_SIZE ||
        nextHead.y < 0 ||
        nextHead.y >= BOARD_SIZE;
      const hitSelf = bodyToCheck.some(
        (segment) => segment.x === nextHead.x && segment.y === nextHead.y
      );

      if (hitWall || hitSelf) {
        return {
          ...current,
          direction: nextDirection,
          queuedDirection: null,
          status: "over",
        };
      }

      const snake = [nextHead, ...current.snake];

      if (!willEat) {
        snake.pop();
      }

      const score = willEat ? current.score + 1 : current.score;

      return {
        ...current,
        direction: nextDirection,
        food: willEat ? spawnFood(snake) : current.food,
        queuedDirection: null,
        score,
        snake,
      };
    });
  });

  const tickLabTraining = useEffectEvent(() => {
    const controller = labRef.current;

    for (let index = 0; index < LAB_BATCH_SIZE; index += 1) {
      trainLabStep(controller);
    }

    syncLabView();
  });

  const tickLabWatch = useEffectEvent(() => {
    watchLabStep(labRef.current);
    syncLabView();
  });

  const handleKeyDown = useEffectEvent((event: KeyboardEvent) => {
    const key = event.key.toLowerCase();

    if (
      key === "arrowup" ||
      key === "arrowdown" ||
      key === "arrowleft" ||
      key === "arrowright" ||
      key === "w" ||
      key === "a" ||
      key === "s" ||
      key === "d" ||
      key === " " ||
      key === "enter"
    ) {
      event.preventDefault();
    }

    if (viewMode !== "manual") {
      return;
    }

    switch (key) {
      case "arrowup":
      case "w":
        steer("up");
        return;
      case "arrowdown":
      case "s":
        steer("down");
        return;
      case "arrowleft":
      case "a":
        steer("left");
        return;
      case "arrowright":
      case "d":
        steer("right");
        return;
      case " ":
        toggleRun();
        return;
      case "enter":
        if (game.status === "paused") {
          toggleRun();
          return;
        }

        startRun(game.direction);
        return;
      default:
        return;
    }
  });

  useEffect(() => {
    if (game.score > persistedBestScore) {
      writeBestScore(game.score);
    }
  }, [game.score, persistedBestScore]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown, { passive: false });

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    if (game.status !== "running") {
      return;
    }

    const intervalId = window.setInterval(stepGame, getStepInterval(game.score));

    return () => {
      window.clearInterval(intervalId);
    };
  }, [game.score, game.status]);

  useEffect(() => {
    if (lab.mode !== "training") {
      return;
    }

    let frameId = 0;

    const loop = () => {
      tickLabTraining();

      if (labRef.current.mode === "training") {
        frameId = window.requestAnimationFrame(loop);
      }
    };

    frameId = window.requestAnimationFrame(loop);

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [lab.mode]);

  useEffect(() => {
    if (lab.mode !== "watching") {
      return;
    }

    const intervalId = window.setInterval(tickLabWatch, LAB_WATCH_INTERVAL);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [lab.mode]);

  const manualToneColor =
    game.status === "running"
      ? "var(--accent-lime)"
      : game.status === "over"
        ? "var(--accent-coral)"
        : "var(--accent-gold)";
  const labToneColor =
    lab.mode === "training"
      ? "var(--accent-cyan)"
      : lab.mode === "watching"
        ? "var(--accent-lime)"
        : "var(--accent-gold)";
  const bestScore = Math.max(game.score, persistedBestScore);
  const pace = `${Math.round(1000 / getStepInterval(game.score))} hz`;
  const primaryAction =
    game.status === "running"
      ? "Pause run"
      : game.status === "paused"
        ? "Resume run"
        : game.status === "over"
          ? "Restart run"
          : "Start run";
  const manualOverlay = game.status === "running" ? null : overlayCopy(game.status);
  const rlOverlay = labOverlayCopy(lab.mode);
  const activeBoard = viewMode === "manual" ? game : lab.board;
  const activeToneColor = viewMode === "manual" ? manualToneColor : labToneColor;
  const activeStatusLabel =
    viewMode === "manual" ? statusLabel(game.status) : labStatusLabel(lab.mode);
  const activeOverlay = viewMode === "manual" ? manualOverlay : rlOverlay;
  const snakeLookup = new Map(
    activeBoard.snake.map((segment, index) => [`${segment.x}:${segment.y}`, index])
  );
  const cells = [];
  const scoreHistory = lab.history.map((entry) => entry.score);
  const rewardHistory = lab.history.map((entry) => entry.reward);
  const qRange = chartRange(lab.qValues);

  for (let y = 0; y < BOARD_SIZE; y += 1) {
    for (let x = 0; x < BOARD_SIZE; x += 1) {
      const key = `${x}:${y}`;
      const snakeIndex = snakeLookup.get(key);
      let className = "arena-cell";

      if (snakeIndex === 0) {
        className += " arena-cell--head";
      } else if (typeof snakeIndex === "number") {
        className += " arena-cell--trail";
      } else if (activeBoard.food.x === x && activeBoard.food.y === y) {
        className += " arena-cell--food";
      }

      cells.push(<div key={key} aria-hidden="true" className={className} />);
    }
  }

  return (
    <section className="dojo-shell mx-auto flex min-h-screen w-full max-w-7xl items-center px-4 py-6 sm:px-6 lg:px-8">
      <div className="glass-panel w-full rounded-[2rem] p-4 sm:p-6 lg:p-8">
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="scanline-text">Playable snake + RL training lab</p>
              <h1 className="mt-3 font-mono text-4xl uppercase tracking-[0.18em] text-white sm:text-5xl lg:text-6xl">
                Shadow Coil
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-6 text-white/70 sm:text-base">
                Run the snake by hand, then switch into the Q-learning lab to
                watch the policy improve episode by episode.
              </p>
            </div>

            <div className="flex flex-col gap-3 sm:items-end">
              <div
                aria-live="polite"
                className="status-chip inline-flex items-center gap-3 self-start rounded-full px-4 py-2 text-xs uppercase tracking-[0.28em] text-white/80 sm:px-5"
              >
                <span
                  className="h-2.5 w-2.5 rounded-full shadow-[0_0_16px_currentColor]"
                  style={{ backgroundColor: activeToneColor, color: activeToneColor }}
                />
                {activeStatusLabel}
              </div>

              <div className="mode-switch inline-flex rounded-full p-1">
                <ModeButton
                  active={viewMode === "manual"}
                  label="Manual"
                  onClick={() => {
                    setViewMode("manual");
                  }}
                />
                <ModeButton
                  active={viewMode === "rl"}
                  label="RL Lab"
                  onClick={() => {
                    setViewMode("rl");
                  }}
                />
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
            <div className="flex flex-col gap-4">
              {viewMode === "manual" ? (
                <div className="grid gap-3 sm:grid-cols-3">
                  <StatCard label="Score" value={formatStat(game.score)} />
                  <StatCard label="Best" value={formatStat(bestScore)} />
                  <StatCard label="Pace" value={pace} />
                </div>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <StatCard label="Episodes" value={lab.episodes.toString()} />
                  <StatCard label="Best AI" value={lab.bestScore.toString()} />
                  <StatCard
                    label="Explore"
                    value={`${Math.round(lab.epsilon * 100)}%`}
                  />
                  <StatCard
                    label="Known"
                    value={lab.knownStates.toString()}
                  />
                </div>
              )}

              <div className="arena-frame relative overflow-hidden rounded-[1.75rem] p-3 sm:p-4">
                <div className="mb-3 flex items-center justify-between gap-4">
                  <span className="scanline-text">
                    Battlefield {BOARD_SIZE} x {BOARD_SIZE}
                  </span>
                  <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-xs uppercase tracking-[0.24em] text-white/60">
                    {activeBoard.snake.length} links
                  </span>
                </div>

                <div className="relative">
                  <div
                    className="arena-grid"
                    style={{
                      gridTemplateColumns: `repeat(${BOARD_SIZE}, minmax(0, 1fr))`,
                    }}
                  >
                    {cells}
                  </div>

                  {activeOverlay ? (
                    <div className="overlay-panel absolute inset-3 flex items-center justify-center rounded-[1.45rem] p-4 text-center sm:inset-4">
                      <div className="max-w-xs">
                        <p className="scanline-text">{activeOverlay.title}</p>
                        <p className="mt-3 font-mono text-2xl uppercase tracking-[0.18em] text-white sm:text-3xl">
                          {activeStatusLabel}
                        </p>
                        <p className="mt-3 text-sm leading-6 text-white/70">
                          {activeOverlay.body}
                        </p>
                        <button
                          className="command-button command-button--primary mt-5 rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]"
                          onClick={() => {
                            if (viewMode === "manual") {
                              if (game.status === "paused") {
                                toggleRun();
                                return;
                              }

                              startRun(game.direction);
                              return;
                            }

                            setLabMode("training");
                          }}
                          type="button"
                        >
                          {activeOverlay.action}
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            {viewMode === "manual" ? (
              <div className="flex flex-col gap-4">
                <div className="info-tile rounded-[1.5rem] p-5">
                  <p className="scanline-text">Control Center</p>
                  <p className="mt-3 text-sm leading-6 text-white/70">
                    Arrow keys and WASD steer the snake. Space toggles the run.
                    Enter starts a fresh pass.
                  </p>

                  <div className="mt-5 grid grid-cols-3 gap-2">
                    <div />
                    <DirectionButton
                      direction="up"
                      glyph="^"
                      label="Up"
                      onPress={steer}
                    />
                    <div />
                    <DirectionButton
                      direction="left"
                      glyph="<"
                      label="Left"
                      onPress={steer}
                    />
                    <div className="info-tile flex min-h-14 items-center justify-center rounded-[1rem] px-2 text-center font-mono text-[0.7rem] uppercase tracking-[0.18em] text-white/55">
                      {statusLabel(game.status)}
                    </div>
                    <DirectionButton
                      direction="right"
                      glyph=">"
                      label="Right"
                      onPress={steer}
                    />
                    <div />
                    <DirectionButton
                      direction="down"
                      glyph="v"
                      label="Down"
                      onPress={steer}
                    />
                    <div />
                  </div>

                  <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
                    <button
                      className="command-button command-button--primary rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]"
                      onClick={toggleRun}
                      type="button"
                    >
                      {primaryAction}
                    </button>
                    <button
                      className="command-button rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-white/80"
                      onClick={resetBoard}
                      type="button"
                    >
                      Reset board
                    </button>
                  </div>
                </div>

                <div className="info-tile rounded-[1.5rem] p-5">
                  <p className="scanline-text">Mission Brief</p>
                  <ul className="mt-4 space-y-3 text-sm leading-6 text-white/74">
                    <li className="flex gap-3">
                      <span
                        className="mt-2 h-2 w-2 shrink-0 rounded-full"
                        style={{ backgroundColor: "var(--accent-lime)" }}
                      />
                      Eat the ember to extend the snake and raise the pace.
                    </li>
                    <li className="flex gap-3">
                      <span
                        className="mt-2 h-2 w-2 shrink-0 rounded-full"
                        style={{ backgroundColor: "var(--accent-cyan)" }}
                      />
                      You can cut tight turns, but reversing direction is blocked.
                    </li>
                    <li className="flex gap-3">
                      <span
                        className="mt-2 h-2 w-2 shrink-0 rounded-full"
                        style={{ backgroundColor: "var(--accent-coral)" }}
                      />
                      Touching the wall or your own trail ends the run.
                    </li>
                  </ul>
                </div>

                <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
                  <MicroCard
                    kicker="Keys"
                    value="W A S D"
                    detail="Alt steering"
                  />
                  <MicroCard
                    kicker="Space"
                    value="Pause"
                    detail="Freeze or resume"
                  />
                  <MicroCard
                    kicker="Enter"
                    value="New Run"
                    detail="Fresh board"
                  />
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                <div className="info-tile rounded-[1.5rem] p-5">
                  <p className="scanline-text">Learning Controls</p>
                  <p className="mt-3 text-sm leading-6 text-white/70">
                    Training runs live in the browser with tabular Q-learning.
                    The agent explores at first, then gradually exploits the
                    best moves it has discovered.
                  </p>

                  <div className="mt-4 grid gap-2">
                    <button
                      className="command-button command-button--primary rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em]"
                      onClick={() => {
                        setLabMode(lab.mode === "training" ? "idle" : "training");
                      }}
                      type="button"
                    >
                      {lab.mode === "training" ? "Pause training" : "Start training"}
                    </button>
                    <button
                      className="command-button rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-white/80"
                      onClick={() => {
                        setLabMode(lab.mode === "watching" ? "idle" : "watching");
                      }}
                      type="button"
                    >
                      {lab.mode === "watching" ? "Stop demo" : "Watch policy"}
                    </button>
                    <button
                      className="command-button rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.16em] text-white/80"
                      onClick={resetLab}
                      type="button"
                    >
                      Reset model
                    </button>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
                    <MicroCard
                      kicker="Avg score"
                      value={lab.averageScore.toFixed(2)}
                      detail="Last 12 episodes"
                    />
                    <MicroCard
                      kicker="Avg reward"
                      value={lab.averageReward.toFixed(2)}
                      detail="Includes shaped rewards"
                    />
                  </div>
                </div>

                <div className="info-tile rounded-[1.5rem] p-5">
                  <p className="scanline-text">Learning Process</p>
                  <p className="mt-3 text-sm leading-6 text-white/70">
                    Q(s,a) = Q + alpha * (reward + gamma * max next - Q)
                  </p>

                  <div className="mt-4 grid gap-3">
                    <LearningChip
                      label="Heading"
                      value={describeHeading(lab.currentObservation.heading)}
                    />
                    <LearningChip
                      label="Food vector"
                      value={`${describeFoodAxis(
                        lab.currentObservation.foodRelation.x,
                        "left",
                        "right"
                      )} / ${describeFoodAxis(
                        lab.currentObservation.foodRelation.y,
                        "up",
                        "down"
                      )}`}
                    />
                    <LearningChip
                      label="Action"
                      value={actionLabel(lab.lastAction)}
                    />
                    <LearningChip label="Update" value={lab.lastOutcome} />
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <StateFlag
                      active={lab.currentObservation.dangers.forward}
                      label="Danger forward"
                    />
                    <StateFlag
                      active={lab.currentObservation.dangers.left}
                      label="Danger left"
                    />
                    <StateFlag
                      active={lab.currentObservation.dangers.right}
                      label="Danger right"
                    />
                  </div>

                  <div className="mt-5 grid gap-3">
                    {AGENT_ACTIONS.map((action, index) => (
                      <QValueMeter
                        key={action}
                        action={action}
                        max={qRange.max}
                        min={qRange.min}
                        value={lab.qValues[index]}
                      />
                    ))}
                  </div>
                </div>

                <div className="grid gap-3">
                  <SparklineCard
                    accent="var(--accent-lime)"
                    label="Episode score"
                    values={scoreHistory}
                  />
                  <SparklineCard
                    accent="var(--accent-cyan)"
                    label="Episode reward"
                    values={rewardHistory}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-tile rounded-[1.35rem] p-4">
      <p className="scanline-text">{label}</p>
      <p className="mt-3 font-mono text-3xl uppercase tracking-[0.16em] text-white sm:text-4xl">
        {value}
      </p>
    </div>
  );
}

function MicroCard({
  detail,
  kicker,
  value,
}: {
  detail: string;
  kicker: string;
  value: string;
}) {
  return (
    <div className="info-tile rounded-[1.2rem] p-4">
      <p className="scanline-text">{kicker}</p>
      <p className="mt-2 font-mono text-xl uppercase tracking-[0.14em] text-white">
        {value}
      </p>
      <p className="mt-2 text-sm text-white/60">{detail}</p>
    </div>
  );
}

function DirectionButton({
  direction,
  glyph,
  label,
  onPress,
}: {
  direction: Direction;
  glyph: string;
  label: string;
  onPress: (direction: Direction) => void;
}) {
  return (
    <button
      aria-label={label}
      className="dpad-button min-h-14 rounded-[1rem] font-mono text-lg uppercase tracking-[0.18em] text-white"
      onClick={() => {
        onPress(direction);
      }}
      type="button"
    >
      {glyph}
    </button>
  );
}

function ModeButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={`mode-button rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] ${
        active ? "mode-button--active" : "text-white/72"
      }`}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  );
}

function LearningChip({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="learning-chip rounded-[1rem] px-3 py-3">
      <p className="scanline-text">{label}</p>
      <p className="mt-2 text-sm leading-5 text-white/78">{value}</p>
    </div>
  );
}

function StateFlag({
  active,
  label,
}: {
  active: boolean;
  label: string;
}) {
  return (
    <span
      className={`state-flag rounded-full px-3 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.18em] ${
        active ? "state-flag--active" : "text-white/60"
      }`}
    >
      {label}
    </span>
  );
}

function QValueMeter({
  action,
  max,
  min,
  value,
}: {
  action: AgentAction;
  max: number;
  min: number;
  value: number;
}) {
  const range = max - min;
  const width =
    range === 0 ? 50 : 12 + ((value - min) / range) * 88;

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between gap-3 text-sm text-white/72">
        <span>{actionLabel(action)}</span>
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-white/54">
          {formatSignedMetric(value)}
        </span>
      </div>
      <div className="q-track h-2.5 rounded-full">
        <div
          className="q-fill h-full rounded-full"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

function SparklineCard({
  accent,
  label,
  values,
}: {
  accent: string;
  label: string;
  values: number[];
}) {
  const width = 260;
  const height = 72;
  const { max, min } = chartRange(values);
  const range = max - min || 1;
  const points =
    values.length > 1
      ? values
          .map((value, index) => {
            const x = (index / (values.length - 1)) * width;
            const y = height - ((value - min) / range) * height;

            return `${x},${y}`;
          })
          .join(" ")
      : "";
  const latest = values.at(-1) ?? 0;

  return (
    <div className="chart-card rounded-[1.35rem] p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="scanline-text">{label}</p>
        <p className="font-mono text-xs uppercase tracking-[0.18em] text-white/56">
          {values.length === 0 ? "No data" : latest.toFixed(2)}
        </p>
      </div>

      <div className="mt-3">
        {values.length > 1 ? (
          <svg
            aria-hidden="true"
            className="h-[4.5rem] w-full"
            preserveAspectRatio="none"
            viewBox={`0 0 ${width} ${height}`}
          >
            <path
              d={`M0 ${height} L${width} ${height}`}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="1"
            />
            <polyline
              fill="none"
              points={points}
              stroke={accent}
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="3"
            />
          </svg>
        ) : (
          <div className="flex h-[4.5rem] items-center justify-center rounded-[1rem] border border-white/8 bg-white/3 text-xs uppercase tracking-[0.2em] text-white/40">
            Start training to populate this chart
          </div>
        )}
      </div>
    </div>
  );
}
