# Hej och välkommen till en SIR-baserad simulerad pandemi, där ett neuralt nätverk försöker begränsa smittspridningen!

För en beskrivning av hur simuleringsmodellen fungerar och vilka parametrar som finns, se avsnitt 2.1 *Modellteori* i slutrapport.pdf.

Det finns två (?) olika varianter av simuleringen. Dessa är anpassade för olika ändamål och beskrivs nedan. Efter att dessa två varianter introducerats genomgås kort hur man gör för att köra en simulering i moln-tjänsten "Google Colab", där programmen lagras (dåligt ordval?).

Det första alternativet är SIR_grund.ipynb [länk?]. Detta är den grundläggande modellen som är bra på att ge en visualisering av smittspridningen grafiskt. Bra om man vill se hur en simulation ser ut, men inte den bästa versionen för att dra slutsatser om hur olika parametrar påverkar smittspridningen. 

Det andra alternativet är SIR_adv.ipynb. Denna version har samma modularitet som grundmodellen, men visar inte upp en grafisk bild av samhället och agenterna. Detta leder till att den kör snabbare. Det finns också möjlighet att medelvärdesbilda över ett antal körningar med samma begynnelsevillkor, för att se till att det resultat man får inte bara var slumpmässigt bra eller dåligt. 

# Hur kör man programmen?  / Hur fungerar Google Colab? 

När du bestämt dig för vilket program du vill köra så klickar du in på motsvarande länk i ovanstående stycken. Då omdirigeras du till Google Colab. Programmet är skrivet i blockform, och för att kunna göra simuleringar behöver vissa kodblock först köras för att initialisera funktionerna som används.  Dessa har markerats med en asterisk, \*. Snabbast är att gå ner till blocket med en trippel-asterisk \*\*\*, markera det blocket. Klicka sedan på *Körning -> Kör Före* i menyfältet på toppen av hemsidan.

Nedanför detta block finns ett formulär med ett antal reglage och kryssrutor som ställer in parametrarna för simuleringen. När parametrarna ställts in efter önskemål, klicka på *Play*- knappen till vänster vid toppen av formulärsblocket. 

-------
Vill vi ha en kort beskrivning här eller skulle det räcka med ovanstående? 
