export declare const LANGUAGES: Readonly<{
    en: "English";
    es: "Español";
    de: "Deutsch";
    ja: "日本語";
    fr: "Français";
    ru: "Русский";
}>;
type Locale = keyof typeof LANGUAGES;
export declare function chooseLocale(preferred?: readonly string[], explicit?: string): Locale;
export declare const currentLocale: () => "de" | "en" | "es" | "fr" | "ja" | "ru";
export declare function t(message: string, values?: Record<string, string | number>): string;
export declare function number(value: number, options?: Intl.NumberFormatOptions): string;
export declare function runtimeText(message: string): string;
export declare function setLanguage(value: string, { persist }?: {
    persist?: boolean | undefined;
}): Promise<void>;
export declare function initializeI18n(): Promise<void>;
export {};
