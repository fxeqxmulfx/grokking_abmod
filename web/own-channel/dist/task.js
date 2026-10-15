import { element as $ } from './dom.js';
import { t, currentLocale } from './i18n.js';
const pages = ['intro', 'task', 'solution', 'examples', 'model'];
const introPages = ['why', 'epoch', 'paper', 'chart', 'weights', 'probe', 'grokking', 'findings'];
const modelViews = ['diagram', 'types', 'equations'];
const examples = [[6, 4, 50], [10, 3, 68], [1, 2, 49], [1, 96, 96], [96, 2, 48], [0, 7, 0], [42, 42, 1], [7, 1, 7]];
export function initializeTask() {
    const dialog = $('task-dialog');
    let page = 0, modelView = 0, introView = 0;
    function updateNavigation() {
        $('task-back').disabled = page === 0 && introView === 0;
        $('task-next').disabled = page === pages.length - 1 && modelView === modelViews.length - 1;
    }
    function selectIntro(index) {
        introView = index;
        dialog.querySelector('.intro-panels').scrollTop = 0;
        for (const [i, name] of introPages.entries()) {
            $(`intro-page-${name}`).hidden = i !== index;
            const button = $(`intro-tab-${name}`);
            button.classList.toggle('active', i === index);
            button.setAttribute('aria-selected', String(i === index));
            button.tabIndex = i === index ? 0 : -1;
        }
        updateNavigation();
    }
    function selectModel(index) {
        modelView = index;
        for (const [i, name] of modelViews.entries()) {
            $(`model-page-${name}`).hidden = i !== modelView;
            const button = $(`model-tab-${name}`);
            button.classList.toggle('active', i === modelView);
            button.setAttribute('aria-selected', String(i === modelView));
            button.tabIndex = i === modelView ? 0 : -1;
        }
        dialog.querySelector('.task-model-note').hidden = modelView !== 0;
        updateNavigation();
    }
    function select(index) {
        page = Math.max(0, Math.min(pages.length - 1, index));
        for (const [i, name] of pages.entries()) {
            $(`task-page-${name}`).hidden = i !== page;
            const button = dialog.querySelector(`[data-task-tab="${name}"]`);
            button.classList.toggle('active', i === page);
            button.setAttribute('aria-selected', String(i === page));
            button.tabIndex = i === page ? 0 : -1;
        }
        updateNavigation();
    }
    function advance(direction) {
        if (page === 0 && introView + direction >= 0 && introView + direction < introPages.length) {
            selectIntro(introView + direction);
        }
        else if (page === pages.length - 1 && modelView + direction >= 0 && modelView + direction < modelViews.length) {
            selectModel(modelView + direction);
        }
        else {
            const next = page + direction;
            if (next < 0 || next >= pages.length)
                return;
            if (next === 0)
                selectIntro(introPages.length - 1);
            if (next === pages.length - 1)
                selectModel(0);
            select(next);
        }
    }
    function localizeDiagrams() {
        for (const source of dialog.querySelectorAll('[data-model-diagram-portrait]')) {
            source.srcset = new URL(`./diagrams/model-${currentLocale()}-portrait.svg`, import.meta.url).href;
        }
        for (const image of dialog.querySelectorAll('[data-task-diagram]')) {
            const model = image.dataset.taskDiagram === 'model';
            image.src = new URL(`./diagrams/${model ? 'model' : 'division'}-${currentLocale()}.svg`, import.meta.url).href;
            image.alt = t(model ? 'Rotate90 model architecture' : 'Mathematical solution');
        }
    }
    for (const [a, b, q] of examples) {
        if (b * q % 97 !== a)
            throw new Error('Invalid modular-division example.');
        const row = document.createElement('tr'), equation = document.createElement('td'), check = document.createElement('td');
        equation.textContent = `${a} / ${b} = ${q}`;
        check.textContent = `${b} × ${q} = ${b * q} = ${(b * q - a) / 97} × 97 + ${a}`;
        row.append(equation, check);
        $('task-example-rows').append(row);
    }
    for (const button of dialog.querySelectorAll('[data-task-tab]'))
        button.onclick = () => select(pages.indexOf(button.dataset.taskTab ?? ''));
    for (const button of dialog.querySelectorAll('[data-intro-tab]'))
        button.onclick = () => selectIntro(introPages.indexOf(button.dataset.introTab ?? ''));
    dialog.querySelector('.intro-tabs').onkeydown = event => {
        const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (!offset && event.key !== 'Home' && event.key !== 'End')
            return;
        event.preventDefault();
        selectIntro(event.key === 'Home' ? 0 : event.key === 'End' ? introPages.length - 1 : (introView + offset + introPages.length) % introPages.length);
        $(`intro-tab-${introPages[introView]}`).focus();
    };
    for (const button of dialog.querySelectorAll('[data-model-explanation]'))
        button.onclick = () => selectModel(modelViews.indexOf(button.dataset.modelExplanation ?? ''));
    dialog.querySelector('.task-model-tabs').onkeydown = event => {
        const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (!offset && event.key !== 'Home' && event.key !== 'End')
            return;
        event.preventDefault();
        selectModel(event.key === 'Home' ? 0 : event.key === 'End' ? modelViews.length - 1 : (modelView + offset + modelViews.length) % modelViews.length);
        $(`model-tab-${modelViews[modelView]}`).focus();
    };
    dialog.querySelector('.task-tabs').onkeydown = event => {
        const offset = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
        if (!offset && event.key !== 'Home' && event.key !== 'End')
            return;
        event.preventDefault();
        select(event.key === 'Home' ? 0 : event.key === 'End' ? pages.length - 1 : (page + offset + pages.length) % pages.length);
        dialog.querySelector(`[data-task-tab="${pages[page]}"]`).focus();
    };
    $('task-back').onclick = () => advance(-1);
    $('task-next').onclick = () => advance(1);
    for (const id of ['close-task', 'task-open'])
        $(id).onclick = () => dialog.close();
    $('task-help').onclick = () => { select(0); selectIntro(0); dialog.showModal(); };
    $('chart-help').onclick = () => { select(0); selectIntro(3); dialog.showModal(); };
    $('scales-help').onclick = () => { select(0); selectIntro(introPages.indexOf('probe')); dialog.showModal(); };
    window.addEventListener('localechange', localizeDiagrams);
    localizeDiagrams();
    selectModel(0);
    selectIntro(0);
    select(0);
    dialog.showModal();
}
//# sourceMappingURL=task.js.map