import { element as $ } from './dom.js';
import { t } from './i18n.js';
import { HELP_TOPICS } from './help-content.js';
export function initializeHelp() {
    const dialog = $('help-dialog');
    const buttons = document.querySelectorAll('button[data-help]');
    let current = null;
    const topic = () => HELP_TOPICS[current];
    function render() {
        if (!current)
            return;
        $('help-title').textContent = t(topic().title);
        const paragraphs = topic().paragraphs.map(source => {
            const paragraph = document.createElement('p');
            paragraph.textContent = t(source);
            return paragraph;
        });
        if (topic().formula) {
            const paragraph = document.createElement('p'), code = document.createElement('code');
            code.textContent = topic().formula;
            paragraph.append(code);
            paragraphs.push(paragraph);
        }
        $('help-content').replaceChildren(...paragraphs);
        $('help-guide').hidden = !topic().guide;
        $('help-content').scrollTop = 0;
    }
    function localizeButtons() {
        for (const button of buttons) {
            const id = button.dataset.help;
            if (!Object.hasOwn(HELP_TOPICS, id))
                throw new Error(`Unknown help topic: ${id}`);
            const label = t('Help: {topic}', { topic: t(HELP_TOPICS[id].title) });
            button.setAttribute('aria-label', label);
            button.title = label;
        }
    }
    for (const button of buttons)
        button.onclick = event => {
            event.preventDefault();
            event.stopPropagation();
            current = button.dataset.help;
            render();
            if (!dialog.open)
                dialog.showModal();
        };
    for (const id of ['help-close', 'help-done'])
        $(id).onclick = () => dialog.close();
    dialog.onclick = event => {
        const rect = dialog.getBoundingClientRect();
        if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom))
            dialog.close();
    };
    $('help-guide').onclick = () => {
        const guide = topic().guide;
        if (!guide)
            return;
        dialog.close();
        $('task-help').click();
        if (guide === 'task' || guide === 'model')
            $(`task-tab-${guide}`).click();
        else
            $(`intro-tab-${guide}`).click();
    };
    window.addEventListener('localechange', () => { localizeButtons(); render(); });
    localizeButtons();
}
//# sourceMappingURL=help.js.map