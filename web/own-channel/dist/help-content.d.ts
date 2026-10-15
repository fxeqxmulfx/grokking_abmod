export type GuidePage = 'epoch' | 'chart' | 'weights' | 'probe' | 'grokking' | 'findings' | 'task' | 'model';
export interface HelpTopic {
    title: string;
    paragraphs: readonly string[];
    formula?: string;
    guide?: GuidePage;
}
export declare const HELP_TOPICS: {
    width: {
        title: string;
        paragraphs: string[];
        guide: "model";
    };
    presets: {
        title: string;
        paragraphs: string[];
        guide: "findings";
    };
    settings: {
        title: string;
        paragraphs: string[];
    };
    lr: {
        title: string;
        paragraphs: string[];
        guide: "findings";
    };
    wd: {
        title: string;
        paragraphs: string[];
        guide: "findings";
    };
    curves: {
        title: string;
        paragraphs: string[];
        guide: "chart";
    };
    'train-accuracy': {
        title: string;
        paragraphs: string[];
        guide: "chart";
    };
    'test-accuracy': {
        title: string;
        paragraphs: string[];
        guide: "chart";
    };
    'train-loss': {
        title: string;
        paragraphs: string[];
        guide: "chart";
    };
    'test-loss': {
        title: string;
        paragraphs: string[];
        guide: "chart";
    };
    training: {
        title: string;
        paragraphs: string[];
    };
    epoch: {
        title: string;
        paragraphs: string[];
        guide: "epoch";
    };
    seed: {
        title: string;
        paragraphs: string[];
    };
    limit: {
        title: string;
        paragraphs: string[];
        guide: "epoch";
    };
    engine: {
        title: string;
        paragraphs: string[];
    };
    capture: {
        title: string;
        paragraphs: string[];
    };
    performance: {
        title: string;
        paragraphs: string[];
    };
    phase: {
        title: string;
        paragraphs: string[];
        guide: "grokking";
    };
    exports: {
        title: string;
        paragraphs: string[];
    };
    network: {
        title: string;
        paragraphs: string[];
        guide: "model";
    };
    'weight-view': {
        title: string;
        paragraphs: string[];
        guide: "weights";
    };
    highlight: {
        title: string;
        paragraphs: string[];
    };
    channel: {
        title: string;
        paragraphs: string[];
        guide: "probe";
    };
    zoom: {
        title: string;
        paragraphs: string[];
    };
    weights: {
        title: string;
        paragraphs: string[];
        guide: "weights";
    };
    embedding: {
        title: string;
        paragraphs: string[];
        guide: "weights";
    };
    dense: {
        title: string;
        paragraphs: string[];
        guide: "weights";
    };
    readout: {
        title: string;
        paragraphs: string[];
        guide: "weights";
    };
    centered: {
        title: string;
        paragraphs: string[];
        formula: string;
        guide: "weights";
    };
    probe: {
        title: string;
        paragraphs: string[];
        guide: "task";
    };
    contributions: {
        title: string;
        paragraphs: string[];
        guide: "probe";
    };
};
export type HelpId = keyof typeof HELP_TOPICS;
