import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import en from './en.json'; //English
import es from './es.json'; //Spanish
import jp from './jp.json'; //Japanese
import zh from './zh.json'; //Mandarin

i18n.use(initReactI18next).init({
  resources: {
    en: {
      translation: en,
    },
    es: {
      translation: es,
    },
    jp: {
      translation: jp,
    },
    zh: {
      translation: zh,
    },
  },
});

const locale = localStorage.getItem('docsgpt-locale') ?? 'en';
i18n.changeLanguage(locale);

export default i18n;
