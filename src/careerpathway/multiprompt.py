PROMPTS = {
    'en': """Given the career history below, suggest {} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.
- Consider the current position and career history, but it is acceptable to suggest a career transition or a new direction. Fill in the Key Skills section with the skills required for the position.
- Ensure each suggestion is realistic and distinct, which can make a positive or negative impact on career growth.

[Career History]:
{}

[Current Profile]:
- Current Position: {}
- Years of Experience: {}
- Current Skills: {}

Format your response as follows for each position:
Position 1: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]
Position 2: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]

Please provide each position in the format above, with each element clearly marked with its label and separated by vertical bars (|). Each position should be on a separate line.""",

    'ae': """يرجى النظر في تاريخك المهني أدناه واقتراح {} وظائف محتملة كخطوات تالية في مسيرتك المهنية. يجب أن يكون كل اقتراح مساراً مختلفاً يبني على هذه التجربة.
- ضع في اعتبارك الوظيفة الحالية وتاريخك المهني، ولكن من المقبول اقتراح تغيير مسار أو اتجاه جديد. قم بتعبئة قسم المهارات الرئيسية بالمهارات المطلوبة لهذا المنصب.
- تأكد من أن كل اقتراح واقعي ومتميز، بحيث يؤثر إيجاباً أو سلباً على النمو المهني.

[تاريخ العمل]:
{}

[الملف الشخصي الحالي]:
- الوظيفة الحالية: {}
- سنوات الخبرة: {}
- المهارات الحالية: {}

يرجى تقديم كل وظيفة بالصيغة التالية:
الوظيفة 1: [المسمى الوظيفي] | الراتب المتوقع: [النطاق] | الخبرة المطلوبة: [عدد السنوات] | المهارات الرئيسية: [المهارات]
الوظيفة 2: [المسمى الوظيفي] | الراتب المتوقع: [النطاق] | الخبرة المطلوبة: [عدد السنوات] | المهارات الرئيسية: [المهارات]

يرجى تقديم كل وظيفة في الصيغة أعلاه مع وضع كل عنصر بوضوح مع تسمية مميزة وفصلها باستخدام الرموز العمودية (|). يجب أن يكون كل منصب في سطر منفصل.""",

    'am': """Նախագծի պատմության հիման վրա առաջարկեք {} տարբեր հնարավոր հաջորդ աշխատանքները: Յուրաքանչյուր առաջարկ պետք է լինի տարբեր կարիերայի ուղղություն, որը հիմնված է այս փորձի վրա.
- Հաշվի առեք ներկայիս դիրքը և կարիերայի պատմությունը, սակայն կարելի է առաջարկել կարիերայի փոփոխություն կամ նոր ուղղություն: Պարտադիր է, որ յուրաքանչյուր առաջարկ հիմնված լինի իրական և տարբեր ճանապարհների վրա, որոնք կարող են դրական կամ բացասական ազդեցություն ունենալ կարիերայի զարգացման վրա:

[Կարիերայի պատմություն]:
{}

[Ներկայիս պրոֆիլը]:
- Ներկայիս պաշտոնը: {}
- Աշխատանքային փորձը: {}
- Ներկայիս հմտությունները: {}

Խնդրում ենք տրամադրել յուրաքանչյուր առաջարկը հետևյալ ձևաչափով՝ յուրաքանչյուրի համար:
Պաշտոն 1: [Պաշտոնների անվանումը] | Սպասվող աշխատավարձը: [Համարը] | Պահանջվող փորձը: [Տարիերի քանակ] | Կարևոր հմտություններ: [Հմտություններ]
Պաշտոն 2: [Պաշտոնների անվանումը] | Սպասվող աշխատավարձը: [Համարը] | Պահանջվող փորձը: [Տարիերի քանակ] | Կարևոր հմտություններ: [Հմտություններ]

Խնդրում ենք տրամադրել յուրաքանչյուր պաշտոնը վերոհիշյալ ձևաչափով, որպեսզի յուրաքանչյուր տարր ունենա իրանշանակում և բաժանվի ուղղահայաց աղյուսակներով (|). Յուրաքանչյուր պաշտոնը պետք է լինի առանձին շարքում.""",

    'at': """Geben Sie anhand der untenstehenden Karrieregeschichte {} verschiedene mögliche nächste Jobpositionen an. Jede Empfehlung sollte eine andere berufliche Richtung sein, die auf dieser Erfahrung aufbaut.
- Berücksichtigen Sie die aktuelle Position und die Karrieregeschichte, aber es ist auch akzeptabel, einen Karrierewechsel oder eine neue Richtung vorzuschlagen. Füllen Sie den Abschnitt "Schlüsselqualifikationen" mit den für die Position erforderlichen Fähigkeiten aus.
- Stellen Sie sicher, dass jede Empfehlung realistisch und eindeutig ist, sodass sie einen positiven oder negativen Einfluss auf das berufliche Wachstum hat.

[Karrieregeschichte]:
{}

[Aktuelles Profil]:
- Aktuelle Position: {}
- Berufserfahrung: {}
- Aktuelle Fähigkeiten: {}

Geben Sie jede Position im folgenden Format an:
Position 1: [Berufsbezeichnung] | Erwartetes Gehalt: [Spanne] | Erforderliche Erfahrung: [Jahre] | Schlüsselqualifikationen: [Fähigkeiten]
Position 2: [Berufsbezeichnung] | Erwartetes Gehalt: [Spanne] | Erforderliche Erfahrung: [Jahre] | Schlüsselqualifikationen: [Fähigkeiten]

Bitte geben Sie jede Position im oben genannten Format an, wobei jedes Element klar mit seinem Label gekennzeichnet und durch senkrechte Striche (|) getrennt ist. Jede Position sollte in einer eigenen Zeile angegeben werden.""",

    'az': """Aşağıdakı karyera tarixinə əsaslanaraq, {} fərqli mümkün növbəti iş mövqeyini təklif edin. Hər bir təklif bu təcrübəyə əsaslanan fərqli bir karyera istiqaməti olmalıdır.
- Cari vəzifə və karyera tarixini nəzərə alın, lakin karyera keçidi və ya yeni istiqamət təklif etmək də mümkündür. Vəzifə üçün tələb olunan bacarıqları "Əsas Bacarıqlar" hissəsində doldurun.
- Hər bir təklifin real və fərqli olduğuna əmin olun ki, bu da karyera inkişafına müsbət və ya mənfi təsir göstərə bilər.

[Karyera Tarixi]:
{}

[Cari Profil]:
- Cari Vəzifə: {}
- İş Təcrübəsi: {}
- Cari Bacarıqlar: {}

Hər mövqeni aşağıdakı formatda təqdim edin:
Vəzifə 1: [Vəzifə adı] | Gözlənilən Maaş: [Aralıq] | Tələb olunan Təcrübə: [İllər] | Əsas Bacarıqlar: [Bacarıqlar]
Vəzifə 2: [Vəzifə adı] | Gözlənilən Maaş: [Aralıq] | Tələb olunan Təcrübə: [İllər] | Əsas Bacarıqlar: [Bacarıqlar]

Xahiş edirik, hər bir mövqeyi yuxarıda göstərilən formatda təqdim edin, hər bir elementi açıq şəkildə etiketləyin və şaquli xətlər (|) ilə ayırın. Hər mövqe ayrı bir sətirdə olmalıdır.""",

    'br': """Dada a história de carreira abaixo, sugira {} diferentes possíveis próximos cargos. Cada sugestão deve ser uma direção de carreira diferente que se baseia nessa experiência.
- Considere o cargo atual e a história da carreira, mas é aceitável sugerir uma transição de carreira ou uma nova direção. Preencha a seção de habilidades principais com as habilidades necessárias para a posição.
- Garanta que cada sugestão seja realista e distinta, o que pode ter um impacto positivo ou negativo no crescimento da carreira.

[Histórico de Carreira]:
{}

[Perfil Atual]:
- Cargo Atual: {}
- Anos de Experiência: {}
- Habilidades Atuais: {}

Formate sua resposta da seguinte maneira para cada cargo:
Cargo 1: [Título do Cargo] | Salário Esperado: [Faixa] | Experiência Necessária: [Anos] | Habilidades Principais: [Habilidades]
Cargo 2: [Título do Cargo] | Salário Esperado: [Faixa] | Experiência Necessária: [Anos] | Habilidades Principais: [Habilidades]

Por favor, forneça cada cargo no formato acima, com cada elemento claramente marcado com sua etiqueta e separado por barras verticais (|). Cada cargo deve ser apresentado em uma linha separada.""",

    'ca': """Donada la història professional següent, suggereix {} possibles següents llocs de treball. Cada suggeriment hauria de ser una direcció diferent de carrera que construeixi sobre aquesta experiència.
- Tingueu en compte la posició actual i la història professional, però també és acceptable suggerir una transició de carrera o una nova direcció. Ompliu la secció de Competències Clau amb les habilitats necessàries per a la posició.
- Assegureu-vos que cada suggeriment sigui realista i distint, i que pugui tenir un impacte positiu o negatiu en el creixement professional.

[Historial professional]:
{}

[Perfil actual]:
- Posició actual: {}
- Anys d'experiència: {}
- Habilitats actuals: {}

Formateu la vostra resposta de la següent manera per a cada posició:
Posició 1: [Títol de la feina] | Sou esperat: [Rang] | Experiència requerida: [Anys] | Competències clau: [Habilitats]
Posició 2: [Títol de la feina] | Sou esperat: [Rang] | Experiència requerida: [Anys] | Competències clau: [Habilitats]

Per favor, proporcioneu cada posició en el format anterior, amb cada element clarament marcat amb la seva etiqueta i separat per barres verticals (|). Cada posició ha de ser a una línia separada.""",

    'eg': """بناءً على تاريخك المهني أدناه، اقترح {} وظائف محتملة كخطوات تالية. يجب أن يكون كل اقتراح هو اتجاه مهني مختلف يعتمد على هذه التجربة.
- ضع في اعتبارك الوظيفة الحالية وتاريخك المهني، ولكن من المقبول اقتراح تحول مهني أو اتجاه جديد. قم بتعبئة قسم المهارات الرئيسية بالمهارات المطلوبة لهذا المنصب.
- تأكد من أن كل اقتراح واقعي ومميز، بحيث يكون له تأثير إيجابي أو سلبي على النمو المهني.

[تاريخ العمل]:
{}

[الملف الشخصي الحالي]:
- الوظيفة الحالية: {}
- سنوات الخبرة: {}
- المهارات الحالية: {}

يرجى تقديم كل وظيفة على النحو التالي:
الوظيفة 1: [المسمى الوظيفي] | الراتب المتوقع: [النطاق] | الخبرة المطلوبة: [عدد السنوات] | المهارات الرئيسية: [المهارات]
الوظيفة 2: [المسمى الوظيفي] | الراتب المتوقع: [النطاق] | الخبرة المطلوبة: [عدد السنوات] | المهارات الرئيسية: [المهارات]

يرجى تقديم كل وظيفة بالصيغة المذكورة أعلاه مع وضع كل عنصر بوضوح مع تسميته وفصلها باستخدام الفواصل الرأسية (|). يجب أن تكون كل وظيفة في سطر منفصل.""",

    'es': """Dado el historial profesional a continuación, sugiere {} diferentes posibles siguientes puestos de trabajo. Cada sugerencia debe ser una dirección de carrera diferente que se base en esta experiencia.
- Ten en cuenta la posición actual y el historial profesional, pero también es aceptable sugerir una transición de carrera o una nueva dirección. Completa la sección de habilidades clave con las habilidades necesarias para el puesto.
- Asegúrate de que cada sugerencia sea realista y distinta, lo que puede tener un impacto positivo o negativo en el crecimiento profesional.

[Historial Profesional]:
{}

[Perfil Actual]:
- Puesto Actual: {}
- Años de Experiencia: {}
- Habilidades Actuales: {}

Formatea tu respuesta de la siguiente manera para cada puesto:
Puesto 1: [Título del Puesto] | Salario Esperado: [Rango] | Experiencia Requerida: [Años] | Habilidades Clave: [Habilidades]
Puesto 2: [Título del Puesto] | Salario Esperado: [Rango] | Experiencia Requerida: [Años] | Habilidades Clave: [Habilidades]

Por favor, proporciona cada puesto en el formato anterior, con cada elemento claramente marcado con su etiqueta y separado por barras verticales (|). Cada puesto debe estar en una línea separada.""",

    'gb': """Given the career history below, suggest {} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.
- Consider the current position and career history, but it is acceptable to suggest a career transition or a new direction. Fill in the Key Skills section with the skills required for the position.
- Ensure each suggestion is realistic and distinct, which can make a positive or negative impact on career growth.

[Career History]:
{}

[Current Profile]:
- Current Position: {}
- Years of Experience: {}
- Current Skills: {}

Format your response as follows for each position:
Position 1: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]
Position 2: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]

Please provide each position in the format above, with each element clearly marked with its label and separated by vertical bars (|). Each position should be on a separate line.""",

    'gl': """Dado o historial profesional a continuación, suxire {} diferentes posibles seguintes postos de traballo. Cada suxestión debe ser unha dirección de carreira diferente que se base na experiencia.
- Ten en conta a posición actual e o historial profesional, pero tamén é aceptable suxerir unha transición de carreira ou unha nova dirección. Completa a sección de habilidades clave coas habilidades necesarias para o posto.
- Asegúrate de que cada suxestión sexa realista e distinta, o que pode ter un impacto positivo ou negativo no crecemento profesional.

[Historial Profesional]:
{}

[Perfil Actual]:
- Posto Actual: {}
- Anos de Experiencia: {}
- Habilidades Actuais: {}

Formatea a túa resposta do seguinte xeito para cada posto:
Posto 1: [Título do Posto] | Salario Esperado: [Rango] | Experiencia Requerida: [Anos] | Habilidades Clave: [Habilidades]
Posto 2: [Título do Posto] | Salario Esperado: [Rango] | Experiencia Requerida: [Anos] | Habilidades Clave: [Habilidades]

Por favor, proporciona cada posto no formato anterior, con cada elemento claramente marcado coa súa etiqueta e separado por barras verticais (|). Cada posto debe estar nunha liña separada.""",

    'il': """Given the career history below, suggest {} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.
- Consider the current position and career history, but it is acceptable to suggest a career transition or a new direction. Fill in the Key Skills section with the skills required for the position.
- Ensure each suggestion is realistic and distinct, which can make a positive or negative impact on career growth.

[Career History]:
{}

[Current Profile]:
- Current Position: {}
- Years of Experience: {}
- Current Skills: {}

Format your response as follows for each position:
Position 1: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]
Position 2: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]

Please provide each position in the format above, with each element clearly marked with its label and separated by vertical bars (|). Each position should be on a separate line.""",

    'ir': """با توجه به تاریخچه شغلی زیر، {} موقعیت شغلی احتمالی بعدی را پیشنهاد دهید. هر پیشنهاد باید یک مسیر شغلی متفاوت باشد که بر اساس این تجربه بنا شده است.
- موقعیت فعلی و تاریخچه شغلی را در نظر بگیرید، اما پیشنهاد تغییر شغل یا مسیر جدید نیز قابل قبول است. بخش مهارت‌های کلیدی را با مهارت‌های لازم برای موقعیت شغلی پر کنید.
- اطمینان حاصل کنید که هر پیشنهاد واقع‌بینانه و متمایز باشد، که می‌تواند تأثیر مثبت یا منفی بر رشد شغلی داشته باشد.

[تاریخچه شغلی]:
{}

[نمایه فعلی]:
- موقعیت فعلی: {}
- سال‌های تجربه: {}
- مهارت‌های فعلی: {}

لطفاً هر موقعیت را در قالب زیر ارائه دهید:
موقعیت 1: [عنوان شغلی] | پیش‌بینی حقوق: [محدوده] | تجربه مورد نیاز: [سال‌ها] | مهارت‌های کلیدی: [مهارت‌ها]
موقعیت 2: [عنوان شغلی] | پیش‌بینی حقوق: [محدوده] | تجربه مورد نیاز: [سال‌ها] | مهارت‌های کلیدی: [مهارت‌ها]

لطفاً هر موقعیت را در قالب بالا با هر عنصر به وضوح برچسب‌گذاری شده و جدا شده توسط خط عمودی (|) ارائه دهید. هر موقعیت باید در یک خط جداگانه باشد.""",

    'is': """Gefin kariærasögun hér að neðan, vinsamlegast leggðu til {} mismunandi möguleg næstu starfstitla. Hver tillaga ætti að vera í mismunandi starfsferilsstefnu sem byggir á þessari reynslu.
- Taktu tillit til núverandi stöðu og starfsferilsins, en það er í lagi að leggja til starfsferilsbreytingu eða nýja stefnu. Fylltu út hæfnislistann með þeim hæfileikum sem nauðsynlegir eru fyrir stöðuna.
- Tryggðu að hver tillaga sé raunhæf og greinileg, sem getur haft jákvæð eða neikvæð áhrif á starfsþróun.

[Karírasaga]:
{}

[Núverandi prófíll]:
- Nútíma staða: {}
- Ára reynslu: {}
- Núverandi færni: {}

Formatuðu svör þín þannig:
Staða 1: [Starfsheiti] | Áætlaður laun: [Svæði] | Nauðsynleg reynsla: [Ár] | Hæfnislisti: [Hæfni]
Staða 2: [Starfsheiti] | Áætlaður laun: [Svæði] | Nauðsynleg reynsla: [Ár] | Hæfnislisti: [Hæfni]

Vinsamlegast veittu hverja stöðu samkvæmt ofangreindri skipun, með hverju þætti vel merkt og aðskilið með lóðréttum strikjum (|). Hver staða ætti að vera á sinni eigin línu.""",
    'me': """Na osnovu sledeće karijere, predložite {} različite moguće sledeće pozicije. Svaki predlog treba da bude različit pravac karijere koji se oslanja na ovo iskustvo.
- Razmislite o trenutnoj poziciji i karijernoj istoriji, ali je prihvatljivo predložiti promenu karijere ili novi pravac. Popunite sekciju Ključne Veštine sa veštinama potrebnim za poziciju.
- Uverite se da je svaki predlog realan i jasan, što može imati pozitivan ili negativan uticaj na profesionalni rast.

[Karijerna Istorija]:
{}

[Trenutni Profil]:
- Trenutna Pozicija: {}
- Godine Iskustva: {}
- Trenutne Veštine: {}

Formatirajte odgovor na sledeći način za svaku poziciju:
Pozicija 1: [Naziv Posla] | Očekivana Plata: [Raspon] | Potrebno Iskustvo: [Godine] | Ključne Veštine: [Veštine]
Pozicija 2: [Naziv Posla] | Očekivana Plata: [Raspon] | Potrebno Iskustvo: [Godine] | Ključne Veštine: [Veštine]

Molimo vas da svaku poziciju pružite u gore navedenom formatu, sa jasno označenim elementima i odvojenim vertikalnim linijama (|). Svaka pozicija treba biti u posebnoj liniji.""",

    'mk': """На основа на следната кариера, предложете {} различни можни следни позиции. Секој предлог треба да биде различен правец на кариера што се базира на ова искуство.
- Размислете за моменталната позиција и кариерата, но е прифатливо да предложите кариера промена или нов правец. Пополнете го делот Клучни Вештини со вештините потребни за позицијата.
- Осигурајте се дека секој предлог е реален и јасен, што може да има позитивен или негативен ефект на професионалниот раст.

[Историја на Кариерата]:
{}

[Тековен Профил]:
- Тековна Позиција: {}
- Години Искуство: {}
- Тековни Вештини: {}

Форматирајте го вашиот одговор на следниот начин за секоја позиција:
Позиција 1: [Име на Позицијата] | Очекувана Плата: [Распон] | Потребно Искуство: [Години] | Клучни Вештини: [Вештини]
Позиција 2: [Име на Позицијата] | Очекувана Плата: [Распон] | Потребно Искуство: [Години] | Клучни Вештини: [Вештини]

Ве молиме да ја доставите секоја позиција во горенаведениот формат, со јасно означени елементи и разделени со вертикални линии (|). Секој статус треба да биде на посебен ред.""",

    'nz': """Given the career history below, suggest {} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.
- Consider the current position and career history, but it is acceptable to suggest a career transition or a new direction. Fill in the Key Skills section with the skills required for the position.
- Ensure each suggestion is realistic and distinct, which can make a positive or negative impact on career growth.

[Career History]:
{}

[Current Profile]:
- Current Position: {}
- Years of Experience: {}
- Current Skills: {}

Format your response as follows for each position:
Position 1: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]
Position 2: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]

Please provide each position in the format above, with each element clearly marked with its label and separated by vertical bars (|). Each position should be on a separate line.""",

    'pf': """Dado o histórico de carreira abaixo, sugira {} diferentes possíveis próximos cargos. Cada sugestão deve ser uma direção de carreira diferente que se baseia nessa experiência.
- Considere o cargo atual e o histórico da carreira, mas é aceitável sugerir uma transição de carreira ou uma nova direção. Preencha a seção de habilidades principais com as habilidades necessárias para a posição.
- Garanta que cada sugestão seja realista e distinta, o que pode ter um impacto positivo ou negativo no crescimento da carreira.

[Histórico de Carreira]:
{}

[Perfil Atual]:
- Cargo Atual: {}
- Anos de Experiência: {}
- Habilidades Atuais: {}

Formate sua resposta da seguinte maneira para cada cargo:
Cargo 1: [Título do Cargo] | Salário Esperado: [Faixa] | Experiência Necessária: [Anos] | Habilidades Principais: [Habilidades]
Cargo 2: [Título do Cargo] | Salário Esperado: [Faixa] | Experiência Necessária: [Anos] | Habilidades Principais: [Habilidades]

Por favor, forneça cada cargo no formato acima, com cada elemento claramente marcado com sua etiqueta e separado por barras verticais (|). Cada cargo deve estar em uma linha separada.""",

    'rs': """Na osnovu sledeće karijere, predložite {} različite moguće sledeće pozicije. Svaki predlog treba da bude različit pravac karijere koji se oslanja na ovo iskustvo.
- Razmislite o trenutnoj poziciji i karijernoj istoriji, ali je prihvatljivo predložiti promenu karijere ili novi pravac. Popunite sekciju Ključne Veštine sa veštinama potrebnim za poziciju.
- Uverite se da je svaki predlog realan i jasan, što može imati pozitivan ili negativan uticaj na profesionalni rast.

[Karijerna Istorija]:
{}

[Trenutni Profil]:
- Trenutna Pozicija: {}
- Godine Iskustva: {}
- Trenutne Veštine: {}

Formatirajte odgovor na sledeći način za svaku poziciju:
Pozicija 1: [Naziv Posla] | Očekivana Plata: [Raspon] | Potrebno Iskustvo: [Godine] | Ključne Veštine: [Veštine]
Pozicija 2: [Naziv Posla] | Očekivana Plata: [Raspon] | Potrebno Iskustvo: [Godine] | Ključne Veštine: [Veštine]

Molimo vas da svaku poziciju pružite u gore navedenom formatu, sa jasno označenim elementima i odvojenim vertikalnim linijama (|). Svaka pozicija treba biti u posebnoj liniji.""",

    'ru': """Учитывая приведенную ниже историю карьеры, предложите {} различные возможные следующие должности. Каждое предложение должно быть направлением карьеры, которое строится на этом опыте.
- Учтите текущую должность и карьерную историю, но также допустимо предложить карьерный переход или новое направление. Заполните раздел "Ключевые навыки" необходимыми для должности навыками.
- Убедитесь, что каждое предложение реально и отчетливо, что может положительно или отрицательно повлиять на карьерный рост.

[История Карьеры]:
{}

[Текущий Профиль]:
- Текущая Должность: {}
- Опыт: {}
- Текущие Навыки: {}

Форматируйте ваш ответ следующим образом для каждой должности:
Должность 1: [Должность] | Ожидаемая Зарплата: [Диапазон] | Требуемый Опыт: [Годы] | Ключевые Навыки: [Навыки]
Должность 2: [Должность] | Ожидаемая Зарплата: [Диапазон] | Требуемый Опыт: [Годы] | Ключевые Навыки: [Навыки]

Пожалуйста, предоставьте каждую должность в указанном выше формате, с ясно отмеченными элементами и разделенными вертикальными чертами (|). Каждая должность должна быть на отдельной строке.""",

    'si': """දක්වා ඇති වෘත්තීය ඉතිහාසය මත, {} වෙනත් හැකි අනාගත රැකියා තනතුරු යෝජනා කරන්න. සෑම යෝජනාවක්ම මෙම අත්දැකීම් මත ඉදිරියට ගොඩනැගුණු වෙනස් වෘත්තීය මාර්ගයක් විය යුතුය.
- වර්තමාන තනතුර සහ වෘත්තීය ඉතිහාසය සලකා බලන්න, නමුත් වෘත්තීය මාරුවක් හෝ නව මාර්ගයක් යෝජනා කිරීමට අවසර දී ඇත. "ප්‍රධාන දක්ෂතා" කොටසෙහි එම තනතුර සඳහා අවශ්‍ය දක්ෂතා පුරවන්න.
- සෑම යෝජනාවක්ම වාසනාවන්ත හෝ ඍණාත්මක ආකාරයකින් වෘත්තීය වර්ධනම අඩංගු කළ හැකි විය යුතුය.

[වෘත්තීය ඉතිහාසය]:
{}

[වර්තමාන පැතිකඩ]:
- වර්තමාන තනතුර: {}
- අත්දැකීම්: {}
- වර්තමාන දක්ෂතා: {}

ඉක්මනින් පිළිතුරු ලබා දෙන්න:
තනතුර 1: [තනතුර] | අපේක්ෂිත වැටුප්: [යතුරු] | අවශ්‍ය අත්දැකීම්: [වර්ෂ] | ප්‍රධාන දක්ෂතා: [දක්ෂතා]
තනතුර 2: [තනතුර] | අපේක්ෂිත වැටුප්: [යතුරු] | අවශ්‍ය අත්දැකීම්: [වර්ෂ] | ප්‍රධාන දක්ෂතා: [දක්ෂතා]

ඉක්මනින් පිළිතුරු ලබා දෙන්න, සෑම තනතුරක්ම ඉහත පෝරමයේ අනුව ලබා දෙන්න, සහ ඒවා නිවැරදිව ලේබල් කරමින් ලොක්‍රෙස් (|) මඟින් වෙනස් කරන්න. සෑම තනතුරක්ම වෙනත් තීරුවක තිබිය යුතුයි.""",

    'tr': """Aşağıdaki kariyer geçmişine göre, {} farklı olası bir sonraki iş pozisyonunu önerin. Her öneri, bu deneyimi temel alarak farklı bir kariyer yönü olmalıdır.
- Mevcut pozisyonu ve kariyer geçmişini göz önünde bulundurun, ancak kariyer değişikliği veya yeni bir yön önermek de kabul edilebilir. Anahtar Yetenekler bölümünü, pozisyon için gerekli becerilerle doldurun.
- Her önerinin gerçekçi ve belirgin olduğundan emin olun, bu da kariyer gelişimi üzerinde olumlu veya olumsuz bir etki yaratabilir.

[Kariyer Geçmişi]:
{}

[Mevcut Profil]:
- Mevcut Pozisyon: {}
- Deneyim Yılı: {}
- Mevcut Yetenekler: {}

Her pozisyon için cevabınızı aşağıdaki gibi biçimlendirin:
Pozisyon 1: [İş Ünvanı] | Beklenen Maaş: [Aralık] | Gerekli Deneyim: [Yıl] | Anahtar Yetenekler: [Yetenekler]
Pozisyon 2: [İş Ünvanı] | Beklenen Maaş: [Aralık] | Gerekli Deneyim: [Yıl] | Anahtar Yetenekler: [Yetenekler]

Lütfen her pozisyonu yukarıdaki formatta, her bir öğeyi etiketleyerek ve dikey çizgiler (|) ile ayırarak belirtin. Her pozisyon ayrı bir satırda olmalıdır.""",

    'us': """Given the career history below, suggest {} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.
- Consider the current position and career history, but it is acceptable to suggest a career transition or a new direction. Fill in the Key Skills section with the skills required for the position.
- Ensure each suggestion is realistic and distinct, which can make a positive or negative impact on career growth.

[Career History]:
{}

[Current Profile]:
- Current Position: {}
- Years of Experience: {}
- Current Skills: {}

Format your response as follows for each position:
Position 1: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]
Position 2: [Job Title] | Expected Salary: [Range] | Required Experience: [Years] | Key Skills: [Skills]

Please provide each position in the format above, with each element clearly marked with its label and separated by vertical bars (|). Each position should be on a separate line.""",

    'jp': """以下のキャリア履歴に基づき、{} 通りの異なる次の職務を提案してください。それぞれの提案は、この経験を基にした異なるキャリアの方向性であるべきです。
- 現在のポジションとキャリア履歴を考慮してくださいが、キャリアチェンジや新しい方向性を提案することも許容されます。ポジションに必要なスキルを「キー・スキル」セクションに記入してください。
- 各提案が現実的かつ明確であることを確認し、キャリア成長にプラスまたはマイナスの影響を与える可能性があることを意識してください。

[キャリア履歴]:
{}

[現在のプロフィール]:
- 現在のポジション: {}
- 経験年数: {}
- 現在のスキル: {}

各ポジションについて、以下の形式で回答してください:
ポジション 1: [職種] | 予想年収: [範囲] | 必要な経験: [年数] | キースキル: [スキル]
ポジション 2: [職種] | 予想年収: [範囲] | 必要な経験: [年数] | キースキル: [スキル]

上記の形式で、各要素を明確にラベル付けし、縦線（|）で区切ってください。各ポジションは別々の行に記入してください。""",

    'kr': """아래의 경력 이력을 바탕으로 {}개의 다른 가능한 다음 직업 포지션을 제안하십시오. 각 제안은 이 경험을 바탕으로 한 다른 경력 방향이어야 합니다.
- 현재 직책과 경력 이력을 고려하되, 직업 전환이나 새로운 방향을 제시하는 것도 허용됩니다. '핵심 역량' 섹션에 해당 직책에 필요한 역량을 작성하십시오.
- 각 제안이 현실적이고 구체적이어야 하며, 이는 경력 성장에 긍정적이거나 부정적인 영향을 미칠 수 있습니다.

[경력 이력]:
{}

[현재 프로필]:
- 현재 직책: {}
- 경력 연수: {}
- 현재 역량: {}

각 직책에 대해 아래와 같은 형식으로 답변을 작성하십시오:
직책 1: [직무명] | 예상 연봉: [범위] | 필요한 경력: [년수] | 핵심 역량: [역량]
직책 2: [직무명] | 예상 연봉: [범위] | 필요한 경력: [년수] | 핵심 역량: [역량]

각 직책을 위 형식에 맞게, 각 요소가 명확히 라벨링되고 수직선(|)으로 구분되도록 제공해 주세요. 각 직책은 별도의 줄에 작성하십시오.""",

    'cn': """根据以下职业历史，建议 {} 个不同的可能下一个职位。每个建议都应是一个基于此经验的不同职业方向。
- 考虑当前职位和职业历史，但也可以建议职业转型或新方向。请在“核心技能”部分填写该职位所需的技能。
- 确保每个建议都是现实的且清晰的，这可能会对职业发展产生积极或消极的影响。

[职业历史]:
{}

[当前简介]:
- 当前职位: {}
- 工作经验: {}
- 当前技能: {}

请按以下格式提供每个职位：
职位 1: [职位名称] | 预期薪资: [范围] | 所需经验: [年数] | 核心技能: [技能]
职位 2: [职位名称] | 预期薪资: [范围] | 所需经验: [年数] | 核心技能: [技能]

请按照上述格式提供每个职位，并确保每个元素都清楚地标注并用竖线（|）分隔。每个职位应单独占一行。"""
}



BASELINE_PROMPTS = {
    'ae': """{}الموقع من انطلاقاً {} محتملة وظيفي تقدم مسارات بتوفير قم، يتكون كل منها من {} خطوات متتالية.

متطلبات التنسيق:
1. عرض {} مناصب مختلفة
2. يجب أن يظهر كل مسار {} خطوات متتالية (الوظيفة 1 → الوظيفة {})
3. تقديم خطوات تقدم وظيفي واضحة
4. سرد المناصب فقط، دون تفاصيل إضافية

نموذج التنسيق:
المسار 1: الخطوة1 → الخطوة2 → .. → الخطوة{} → الخطوة{}
المسار 2: الخطوة1 → الخطوة2 → .. → الخطوة{} → الخطوة{}

الرجاء عدم كسر الأسطر في مسار واحد. فقط اذكر المناصب في سطر واحد.""",

    'am': """Սկսած {} դիրքից, խնդրում ենք տրամադրել {} հնարավոր կարիերայի զարգացման ուղիներ, յուրաքանչյուրը բաղկացած {} հաջորդական քայլերից։

Ձևաչափի պահանջներ.
1. Ցույց տալ {} տարբեր պաշտոններ
2. Յուրաքանչյուր ուղի պետք է ցույց տա {} հաջորդական քայլեր (Աշխատանք 1 → Աշխատանք {})
3. Ներկայացնել որպես հստակ կարիերայի առաջընթացի քայլեր
4. Թվարկել միայն պաշտոնները՝ առանց լրացուցիչ մանրամասների

Ձևաչափի օրինակ.
Ուղի 1. Քայլ1 → Քայլ2 → .. → Քայլ{} → Քայլ{}
Ուղի 2. Քայլ1 → Քայլ2 → .. → Քայլ{} → Քայլ{}

Խնդրում ենք չկոտրել տողերը մեկ ուղու մեջ։ Պարզապես թվարկեք պաշտոնները մեկ տողում։""",

    'at': """Ausgehend von der Position {}, bitte {} mögliche Karriereentwicklungspfade angeben, die jeweils aus {} aufeinanderfolgenden Schritten bestehen.

Formatanforderungen:
1. Zeige {} verschiedene Positionen
2. Jeder Pfad sollte {} aufeinanderfolgende Schritte zeigen (Job 1 → Job {})
3. Als klare Karriereentwicklungsschritte darstellen
4. Nur Positionen auflisten, ohne zusätzliche Details

Beispielformat:
Pfad 1: Schritt1 → Schritt2 → .. → Schritt{} → Schritt{}
Pfad 2: Schritt1 → Schritt2 → .. → Schritt{} → Schritt{}

Bitte keine Zeilenumbrüche in einem Pfad. Einfach die Positionen in einer Zeile auflisten.""",

    'az': """{} mövqeyindən başlayaraq, hər biri {} ardıcıl addımdan ibarət olan {} potensial karyera inkişaf yolunu təmin edin.

Format tələbləri:
1. {} fərqli vəzifə göstərin
2. Hər yol {} ardıcıl addımı göstərməlidir (İş 1 → İş {})
3. Aydın karyera inkişaf addımları kimi təqdim edin
4. Yalnız vəzifələri sadalayın, əlavə təfərrüatlar olmadan

Format nümunəsi:
Yol 1: Addım1 → Addım2 → .. → Addım{} → Addım{}
Yol 2: Addım1 → Addım2 → .. → Addım{} → Addım{}

Zəhmət olmasa bir yolda sətirləri sındırmayın. Sadəcə vəzifələri bir sətirdə sadalayın.""",

    'br': """A partir da posição de {}, forneça {} possíveis caminhos de progressão na carreira, cada um consistindo em {} etapas sequenciais.

Requisitos de formato:
1. Mostrar {} posições diferentes
2. Cada caminho deve mostrar {} etapas sequenciais (Cargo 1 → Cargo {})
3. Apresentar como etapas claras de progressão na carreira
4. Listar apenas posições, sem detalhes adicionais

Exemplo de formato:
Caminho 1: Etapa1 → Etapa2 → .. → Etapa{} → Etapa{}
Caminho 2: Etapa1 → Etapa2 → .. → Etapa{} → Etapa{}

Por favor, não quebre as linhas em um caminho. Apenas liste as posições em uma linha.""",

    'ca': """Starting from the position of {}, please provide {} potential career progression paths, each consisting of {} sequential steps.

Format requirements:
1. Show {} different positions
2. Each path should show {} sequential steps (Job 1 → Job {})
3. Present as clear career progression steps
4. List positions only, without additional details

Example format:
Path 1: Step1 → Step2 → .. → Step{} → Step{}
Path 2: Step1 → Step2 → .. → Step{} → Step{}

Please don't break the lines in one path. Just list the positions in one line.""",

    'eg': """بدءاً من منصب {}، يرجى تقديم {} مسارات محتملة للتقدم الوظيفي، يتكون كل منها من {} خطوات متتالية.

متطلبات التنسيق:
1. عرض {} مناصب مختلفة
2. يجب أن يظهر كل مسار {} خطوات متتالية (الوظيفة 1 → الوظيفة {})
3. تقديم خطوات تقدم وظيفي واضحة
4. سرد المناصب فقط، دون تفاصيل إضافية

نموذج التنسيق:
المسار 1: الخطوة1 → الخطوة2 → .. → الخطوة{} → الخطوة{}
المسار 2: الخطوة1 → الخطوة2 → .. → الخطوة{} → الخطوة{}

الرجاء عدم كسر الأسطر في مسار واحد. فقط اذكر المناصب في سطر واحد.""",

    'es': """Partiendo de la posición de {}, proporcione {} posibles rutas de progresión profesional, cada una consistiendo en {} pasos secuenciales.

Requisitos de formato:
1. Mostrar {} posiciones diferentes
2. Cada ruta debe mostrar {} pasos secuenciales (Trabajo 1 → Trabajo {})
3. Presentar como pasos claros de progresión profesional
4. Listar solo posiciones, sin detalles adicionales

Formato de ejemplo:
Ruta 1: Paso1 → Paso2 → .. → Paso{} → Paso{}
Ruta 2: Paso1 → Paso2 → .. → Paso{} → Paso{}

Por favor, no corte las líneas en una ruta. Simplemente liste las posiciones en una línea.""",

    'gb': """Starting from the position of {}, please provide {} potential career progression paths, each consisting of {} sequential steps.

Format requirements:
1. Show {} different positions
2. Each path should show {} sequential steps (Job 1 → Job {})
3. Present as clear career progression steps
4. List positions only, without additional details

Example format:
Path 1: Step1 → Step2 → .. → Step{} → Step{}
Path 2: Step1 → Step2 → .. → Step{} → Step{}

Please don't break the lines in one path. Just list the positions in one line.""",

    'gl': """Partindo da posición de {}, proporcione {} posibles rutas de progresión profesional, cada unha consistindo en {} pasos secuenciais.

Requisitos de formato:
1. Mostrar {} posicións diferentes
2. Cada ruta debe mostrar {} pasos secuenciais (Traballo 1 → Traballo {})
3. Presentar como pasos claros de progresión profesional
4. Listar só posicións, sen detalles adicionais

Formato de exemplo:
Ruta 1: Paso1 → Paso2 → .. → Paso{} → Paso{}
Ruta 2: Paso1 → Paso2 → .. → Paso{} → Paso{}

Por favor, non corte as liñas nunha ruta. Simplemente liste as posicións nunha liña.""",

    'il': """החל מהתפקיד של {}, אנא ספק {} מסלולי התקדמות קריירה פוטנציאליים, כל אחד מורכב מ-{} שלבים רצופים.

דרישות פורמט:
1. הצג {} תפקידים שונים
2. כל מסלול צריך להראות {} שלבים רצופים (משרה 1 → משרה {})
3. הצג כשלבי התקדמות קריירה ברורים
4. רשום תפקידים בלבד, ללא פרטים נוספים

פורמט לדוגמה:
מסלול 1: שלב1 → שלב2 → .. → שלב{} → שלב{}
מסלול 2: שלב1 → שלב2 → .. → שלב{} → שלב{}

אנא אל תשבור שורות במסלול אחד. פשוט רשום את התפקידים בשורה אחת.""",

    'ir': """با شروع از موقعیت {}، لطفاً {} مسیر پیشرفت شغلی احتمالی را ارائه دهید که هر کدام شامل {} مرحله متوالی باشد.

الزامات قالب‌بندی:
1. نمایش {} موقعیت مختلف
2. هر مسیر باید {} مرحله متوالی را نشان دهد (شغل 1 → شغل {})
3. ارائه به عنوان مراحل پیشرفت شغلی واضح
4. فقط فهرست موقعیت‌ها، بدون جزئیات اضافی

قالب نمونه:
مسیر 1: مرحله1 → مرحله2 → .. → مرحله{} → مرحله{}
مسیر 2: مرحله1 → مرحله2 → .. → مرحله{} → مرحله{}

لطفاً خطوط را در یک مسیر نشکنید. فقط موقعیت‌ها را در یک خط فهرست کنید.""",

    'is': """Út frá stöðu {}, vinsamlegast gefðu upp {} möguleg starfsframvinduferli, hvert um sig samanstendur af {} röðuðum skrefum.

Sniðkröfur:
1. Sýna {} mismunandi stöður
2. Hvert ferli ætti að sýna {} röðuð skref (Starf 1 → Starf {})
3. Setja fram sem skýr starfsframvinduskref
4. Lista aðeins stöður, án viðbótarupplýsinga

Dæmi um snið:
Leið 1: Skref1 → Skref2 → .. → Skref{} → Skref{}
Leið 2: Skref1 → Skref2 → .. → Skref{} → Skref{}

Vinsamlegast brjótið ekki línur í einni leið. Listið aðeins stöður í einni línu.""",

    'me': """Polazeći od pozicije {}, molimo navedite {} potencijalnih putanja napredovanja u karijeri, od kojih se svaka sastoji od {} uzastopnih koraka.

Zahtjevi za format:
1. Prikazati {} različitih pozicija
2. Svaka putanja treba pokazati {} uzastopnih koraka (Posao 1 → Posao {})
3. Predstaviti kao jasne korake napredovanja u karijeri
4. Navesti samo pozicije, bez dodatnih detalja

Format primjera:
Putanja 1: Korak1 → Korak2 → .. → Korak{} → Korak{}
Putanja 2: Korak1 → Korak2 → .. → Korak{} → Korak{}

Molimo ne lomite redove u jednoj putanji. Samo navedite pozicije u jednom redu.""",

    'mk': """Тргнувајќи од позицијата на {}, ве молиме обезбедете {} потенцијални патеки за напредување во кариерата, секоја составена од {} последователни чекори.

Барања за формат:
1. Прикажете {} различни позиции
2. Секоја патека треба да прикаже {} последователни чекори (Работа 1 → Работа {})
3. Претставете како јасни чекори за напредување во кариерата
4. Наведете само позиции, без дополнителни детали

Пример формат:
Патека 1: Чекор1 → Чекор2 → .. → Чекор{} → Чекор{}
Патека 2: Чекор1 → Чекор2 → .. → Чекор{} → Чекор{}

Ве молиме не кршете ги линиите во една патека. Само наведете ги позициите во една линија.""",

    'nz': """Starting from the position of {}, please provide {} potential career progression paths, each consisting of {} sequential steps.

Format requirements:
1. Show {} different positions
2. Each path should show {} sequential steps (Job 1 → Job {})
3. Present as clear career progression steps
4. List positions only, without additional details

Example format:
Path 1: Step1 → Step2 → .. → Step{} → Step{}
Path 2: Step1 → Step2 → .. → Step{} → Step{}

Please don't break the lines in one path. Just list the positions in one line.""",

    'pf': """En partant du poste de {}, veuillez fournir {} chemins potentiels de progression de carrière, chacun composé de {} étapes séquentielles.

Exigences de format :
1. Montrer {} postes différents
2. Chaque chemin doit montrer {} étapes séquentielles (Poste 1 → Poste {})
3. Présenter comme des étapes claires de progression de carrière
4. Lister uniquement les postes, sans détails supplémentaires

Format d'exemple :
Chemin 1 : Étape1 → Étape2 → .. → Étape{} → Étape{}
Chemin 2 : Étape1 → Étape2 → .. → Étape{} → Étape{}

Veuillez ne pas couper les lignes dans un chemin. Listez simplement les postes sur une ligne.""",

    'rs': """Почевши од позиције {}, молимо вас да обезбедите {} потенцијалних путања напредовања у каријери, од којих се свака састоји од {} узастопних корака.

Захтеви формата:
1. Приказати {} различитих позиција
2. Свака путања треба да прикаже {} узастопних корака (Посао 1 → Посао {})
3. Представити као јасне кораке напредовања у каријери
4. Навести само позиције, без додатних детаља

Пример формата:
Путања 1: Корак1 → Корак2 → .. → Корак{} → Корак{}
Путања 2: Корак1 → Корак2 → .. → Корак{} → Корак{}

Молимо вас да не преламате редове у једној путањи. Само наведите позиције у једном реду.""",

    'ru': """Начиная с позиции {}, пожалуйста, предоставьте {} возможных путей развития карьеры, каждый из которых состоит из {} последовательных шагов.

Требования к формату:
1. Показать {} разных позиций
2. Каждый путь должен показывать {} последовательных шагов (Работа 1 → Работа {})
3. Представить как четкие шаги карьерного роста
4. Перечислить только позиции, без дополнительных деталей

Пример формата:
Путь 1: Шаг1 → Шаг2 → .. → Шаг{} → Шаг{}
Путь 2: Шаг1 → Шаг2 → .. → Шаг{} → Шаг{}

Пожалуйста, не разрывайте строки в одном пути. Просто перечислите позиции в одной строке.""",

    'si': """Izhajajoč iz položaja {}, prosimo, navedite {} možnih poti napredovanja v karieri, vsaka sestavljena iz {} zaporednih korakov.

Zahteve za format:
1. Prikažite {} različnih položajev
2. Vsaka pot naj prikaže {} zaporednih korakov (Delo 1 → Delo {})
3. Predstavite kot jasne korake napredovanja v karieri
4. Navedite samo položaje, brez dodatnih podrobnosti

Primer formata:
Pot 1: Korak1 → Korak2 → .. → Korak{} → Korak{}
Pot 2: Korak1 → Korak2 → .. → Korak{} → Korak{}

Prosimo, ne lomite vrstic v eni poti. Samo navedite položaje v eni vrstici.""",

    'tr': """{}pozisyonundan başlayarak, her biri {} ardışık adımdan oluşan {} potansiyel kariyer ilerleme yolu sağlayın.

Format gereksinimleri:
1. {} farklı pozisyon gösterin
2. Her yol {} ardışık adım göstermelidir (İş 1 → İş {})
3. Net kariyer ilerleme adımları olarak sunun
4. Sadece pozisyonları listeleyin, ek detaylar olmadan

Örnek format:
Yol 1: Adım1 → Adım2 → .. → Adım{} → Adım{}
Yol 2: Adım1 → Adım2 → .. → Adım{} → Adım{}

Lütfen bir yoldaki satırları bölmeyin. Sadece pozisyonları tek bir satırda listeleyin.""",

    'us': """Starting from the position of {}, please provide {} potential career progression paths, each consisting of {} sequential steps.

Format requirements:
1. Show {} different positions
2. Each path should show {} sequential steps (Job 1 → Job {})
3. Present as clear career progression steps
4. List positions only, without additional details

Example format:
Path 1: Step1 → Step2 → .. → Step{} → Step{}
Path 2: Step1 → Step2 → .. → Step{} → Step{}

Please don't break the lines in one path. Just list the positions in one line.""",

    'jp': """{}のポジションから始めて、{}個の潜在的なキャリア進展経路を提供してください。各経路は{}個の連続したステップで構成されます。

フォーマット要件：
1. {}個の異なるポジションを表示
2. 各経路は{}個の連続したステップを表示（職位1 → 職位{}）
3. 明確なキャリア進展ステップとして提示
4. 追加の詳細なしで、ポジションのみをリスト

フォーマット例：
経路1：ステップ1 → ステップ2 → .. → ステップ{} → ステップ{}
経路2：ステップ1 → ステップ2 → .. → ステップ{} → ステップ{}

1つの経路で改行しないでください。1行でポジションをリストしてください。""",

    'kr': """{} 직책에서 시작하여, {} 개의 잠재적 경력 발전 경로를 제공해주세요. 각 경로는 {} 개의 연속적인 단계로 구성됩니다.

형식 요구사항:
1. {} 개의 서로 다른 직책 표시
2. 각 경로는 {} 개의 연속적인 단계를 보여야 함 (직무 1 → 직무 {})
3. 명확한 경력 발전 단계로 제시
4. 추가 세부사항 없이 직책만 나열

예시 형식:
경로 1: 단계1 → 단계2 → .. → 단계{} → 단계{}
경로 2: 단계1 → 단계2 → .. → 단계{} → 단계{}

하나의 경로에서 줄바꿈을 하지 마세요. 한 줄에 직책들을 나열해주세요.""",

    'cn': """从{}职位开始，请提供{}个潜在的职业发展路径，每个路径由{}个连续步骤组成。

格式要求：
1. 显示{}个不同的职位
2. 每个路径应显示{}个连续步骤（职位1 → 职位{}）
3. 以清晰的职业发展步骤呈现
4. 仅列出职位，不含其他细节

格式示例：
路径1：步骤1 → 步骤2 → .. → 步骤{} → 步骤{}
路径2：步骤1 → 步骤2 → .. → 步骤{} → 步骤{}

请不要在一个路径中换行。只需在一行中列出职位。"""
}