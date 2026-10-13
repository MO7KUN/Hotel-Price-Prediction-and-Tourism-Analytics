/* Pure inference shared with the regression tests. Match sklearn float32 inputs. */
function forestPrice(trees, values) {
  const input = values.map(Math.fround);
  const average = trees.reduce((sum, tree) => {
    let node = 0;
    while (tree.left[node] !== -1) node = input[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    return sum + tree.value[node];
  }, 0) / trees.length;
  return Math.max(Math.expm1(average), 0);
}
if (typeof module !== 'undefined') module.exports = {forestPrice};
if (typeof document !== 'undefined') {
  const el = id => document.getElementById(id), data = window.HOTEL_DESTINATIONS;
  const key = value => value.trim().toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/\s+/g, ' ');
  const cityKey = () => key(data.destinations.aliases[key(el('city').value)] || el('city').value);
  let records = [];
  function resetResult() {
    el('amount').textContent = 'Choose your hotel details'; el('amount').className = 'amount empty';
    el('metrics').replaceChildren(); el('metrics').hidden = true;
  }
  function configureScope() {
    resetResult();
    const record = records.find(r => r.id === el('scope').value);
    el('predict').disabled = !record;
    for (const id of ['adults','distance','reviews','amenities']) el(id).disabled = !record;
    el('status').textContent = record ? 'Historical model available' : 'Awaiting price observations';
    el('source').textContent = '';
    if (!record) {
      el('amount').textContent = 'Price data needed';
      el('explanation').textContent = 'This destination has no trained room-price model yet. Observed hotel prices are needed before we can give an estimate for this city.';
      el('scopeHint').textContent = 'No sufficiently populated stays are available. The Barcelona model is limited to its own market.';
      return;
    }
    el('scopeHint').textContent = `${record.rooms} room(s) · ${record.children} child(ren)${record.child_ages ? ' · ages ' + record.child_ages : ''} · ${record.currency}. Only this observed stay is supported.`;
    el('explanation').textContent = 'Enter hotel details within the observed ranges below, then estimate. This model was evaluated on hotels excluded from training.';
    el('adults').replaceChildren(...record.adults.map(n => new Option(n, n)));
    for (const [id, feature, inverse] of [['distance','distance_center_km',x=>x],['reviews','log_review_count',x=>Math.round(Math.expm1(x))],['amenities','amenity_mentions_count',x=>x]]) {
      const [low, high] = record.domains[feature].map(inverse);
      el(id).min = low; el(id).max = high; el(id).value = Math.max(low, Math.min(high, Number(el(id).value)));
      el(id).title = `Observed range: ${low} to ${high}`;
    }
    el('source').textContent = `Observed sample: ${record.quotes} quotes from ${record.properties} establishments. Held-out test: ${record.test_properties} establishments. Currency: ${record.currency}.`;
  }
  function updateCity() {
    records = data.coverage.filter(r => r.country === el('country').value && r.city_key === cityKey() && r.status === 'ready');
    el('destinationTitle').textContent = `${el('city').value.trim() || 'Choose a city'}, ${el('country').selectedOptions[0].textContent}`;
    el('scope').replaceChildren(...(records.length ? records.map(r=>new Option(`${r.checkin} → ${r.checkout}`,r.id)) : [new Option('No observed price coverage','')]));
    configureScope();
  }
  function updateCountry() {
    const cities = el('country').value === 'MA' ? data.destinations.morocco_shortcuts : data.destinations.global.filter(d=>d.country===el('country').value).map(d=>d.city);
    el('cities').replaceChildren(...cities.map(c=>new Option(c,c)));
    el('city').value = el('country').value === 'MA' ? 'Marrakech' : cities[0];
    updateCity();
  }
  el('country').addEventListener('change',updateCountry); el('city').addEventListener('input',updateCity); el('scope').addEventListener('change',configureScope);
  for (const id of ['adults','distance','reviews','amenities']) el(id).addEventListener('input', () => { if (records.length) resetResult(); });
  for (const [id, destinations] of [['moroccoChips', ['Marrakech','Casablanca','Rabat','Fes','Tangier','Agadir','Essaouira','Oujda','Tetouan','Kenitra','Ouarzazate','Dakhla'].map(city=>({country:'MA',city}))], ['globalChips', data.destinations.global]]) {
    for (const d of destinations) {const button=document.createElement('button');button.type='button';button.textContent=d.city;button.addEventListener('click',()=>{el('country').value=d.country;updateCountry();el('city').value=d.city;updateCity();el('city').focus();});el(id).append(button);}
  }
  el('predictionForm').addEventListener('submit',event=>{
    event.preventDefault(); const record = records.find(r=>r.id===el('scope').value); if(!record) return;
    const values=[Number(el('adults').value),Number(el('distance').value),Math.log1p(Number(el('reviews').value)),Number(el('amenities').value)];
    if(values.some((value,i)=>!Number.isFinite(value)||value<record.domains[data.features[i]][0]-1e-12||value>record.domains[data.features[i]][1]+1e-12)) {el('explanation').textContent='Enter values within this model’s observed ranges.';return;}
    const price = forestPrice(data.trees[record.id], values);
    const money = value => new Intl.NumberFormat('en',{style:'currency',currency:record.currency}).format(value);
    el('amount').className='amount';el('amount').textContent=money(price)+' / night';
    el('explanation').textContent=`Estimated room price for ${record.checkin} → ${record.checkout}. Historical snapshot only. Test error is an average, not an individual confidence interval.`;
    el('metrics').hidden=false;el('metrics').replaceChildren();
    for(const [label,value] of [['Test MAE',money(record.mae)],['Median baseline MAE',money(record.baseline_mae)]]) {const div=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;div.append(strong,span);el('metrics').append(div);}
  });
  updateCountry();
}
