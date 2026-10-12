# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 22:44:59 2026

@author: jmloe
"""

"""Terminal sencilla del articulo: ejecutar con python terminal_binomial.py.
Instalacion (una vez): python -m pip install numpy pandas scipy matplotlib yfinance pandas_market_calendars requests
Solo consola y ventanas matplotlib. Datos reales; no hay modo de datos inventados.
"""
from pathlib import Path
from io import StringIO
import json
import re
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import norm

N = 500                         # Articulo: promedio de N y N+1.
MODELOS = ['AR2-GARCH-t', 'Media-GARCH-t', 'AR2-GARCH-normal', 'Student constante']
GRIEGAS = ['delta', 'gamma', 'vega_1pp', 'theta_dia', 'rho_1pp']
PRECIOS = ['Adapt_hist', 'CRR_hist', 'Adapt_cal', 'CRR_cal']
BASE = Path(__file__).resolve().parent / 'resultados_binomial'
AJUSTES = {}                     # Reutiliza la MLE solo si los retornos son identicos.



def estados(p, R, v0):
    mu, p1, p2, w, a, b, nu = p
    m = np.full(len(R), mu)
    m[2:] += p1 * R[1:-1] + p2 * R[:-2]
    e = R - m
    h = np.full(len(R), v0)
    h[2:] = lfilter([1], [1, -b], w + a * e[1:-1]**2, zi=[b * v0])[0]
    return m, e, h


def perdidas(p, R, v0, normal=False):
    m, e, h = estados(p, R, v0)
    nu = p[-1]
    if normal:
        nll = .5 * (np.log(2 * np.pi * h) + e**2 / h)
    else:
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2))
        nll = -c + .5 * np.log(h) + (nu + 1) / 2 * np.log1p(e**2 / (h * (nu - 2)))
    return np.column_stack((nll, np.log(h + m*m) + R*R / (h + m*m)))


def ajustar(R, modelo=MODELOS[0]):
    x = np.asarray(R) * 100      # Solo escala numerica; se devuelve en retornos decimales.
    clave = (modelo, x.tobytes())
    if clave in AJUSTES:
        return AJUSTES[clave].copy()
    v = np.var(x, ddof=1)
    limites = [(None, None), (-2, 2), (-.999999, .999999), (1e-10, None), (0, 1), (0, 1), (2.000001, None)]
    if modelo in [MODELOS[1], MODELOS[3]]:
        limites[1:3] = [(0, 0), (0, 0)]
    if modelo == MODELOS[3]:
        limites[4:6] = [(0, 0), (0, 0)]
    if modelo == MODELOS[2]:
        limites[-1] = (8, 8)   # nu no interviene en la densidad normal.
    restriccion = lambda p: np.array([.999999-p[4]-p[5], .999999-p[1]-p[2], .999999-p[2]+p[1]])
    soluciones = []
    for a, b, nu in [(.05, .90, 8), (.12, .75, 5), (.03, .95, 20)]:
        if modelo == MODELOS[3]:
            a, b = 0, 0
        p0 = [x.mean(), 0, 0, v*(1-a-b), a, b, 8 if modelo == MODELOS[2] else nu]
        opt = minimize(lambda p: perdidas(p, x, v, modelo == MODELOS[2])[2:, 0].sum(), p0,
                       method='SLSQP', bounds=limites, constraints={'type': 'ineq', 'fun': restriccion},
                       options={'maxiter': 1500, 'ftol': 1e-8})
        if opt.success and np.isfinite(opt.fun) and min(restriccion(opt.x)) >= -1e-8:
            soluciones.append(opt)
    if not soluciones:
        raise ValueError('La estimacion no converge: ' + modelo)
    opt = min(soluciones, key=lambda z: z.fun)
    p = opt.x.copy()
    p[0] /= 100
    p[3] /= 10000
    AJUSTES[clave] = p.copy()
    return p


def malla(T, r, q, curva, n):
    t, A = curva
    if T <= 0 or T > t[-1] + 1e-12:
        raise ValueError('Plazo fuera de la curva de varianza.')
    total = np.interp(T, t, A)
    paso = np.sqrt(total / n)
    fechas = np.interp(np.linspace(0, total, n+1), A, t)
    dt = np.diff(fechas)
    prob = np.expm1((r-q)*dt + paso) / np.expm1(2*paso)
    if not np.all(np.isfinite(prob)) or np.any((prob <= 0) | (prob >= 1)):
        raise ValueError('Malla inadmisible: no cumple d < exp((r-q)dt) < u.')
    return paso, fechas, prob


def arbol(S, K, signo, T, r, q, curva, n=N, americana=True):
    K, signo = np.atleast_1d(K), np.atleast_1d(signo)
    valores = []
    for pasos in [n, n+1]:
        a, fechas, p = malla(T, r, q, curva, pasos)
        precios = S * np.exp(a * (2*np.arange(pasos+1)-pasos))
        V = np.maximum(signo * (precios[:, None]-K), 0)
        for k in range(pasos-1, -1, -1):
            V = np.exp(-r*(fechas[k+1]-fechas[k])) * ((1-p[k])*V[:-1] + p[k]*V[1:])
            if americana:
                precios = precios[:-1] * np.exp(a)
                V = np.maximum(V, signo*(precios[:, None]-K))
        valores.append(V[0])
    return (valores[0]+valores[1])/2


def bs(S, K, signo, T, r, q, varianza):
    v = np.sqrt(varianza)
    d1 = (np.log(S/K)+(r-q)*T+varianza/2)/v
    return signo*(S*np.exp(-q*T)*norm.cdf(signo*d1)-K*np.exp(-r*T)*norm.cdf(signo*(d1-v)))


def sensibilidad(S, K, signo, T, r, q, curva):
    t, A = curva
    sigma = np.sqrt(np.interp(T, t, A)/T)
    hS, hv, hr = .01*S, min(.001, sigma/4), .0001
    V = lambda s=S, plazo=T, tasa=r, factor=1: arbol(s, K, signo, plazo, tasa, q, (t, A*factor)).squeeze()
    v, arriba, abajo = V(), V(s=S+hS), V(s=S-hS)
    return [(arriba-abajo)/(2*hS), (arriba-2*v+abajo)/hS**2,
            .01*(V(factor=((sigma+hv)/sigma)**2)-V(factor=((sigma-hv)/sigma)**2))/(2*hv),
            (V(plazo=T-1/365)-V(plazo=T+1/365))/2,
            .01*(V(tasa=r+hr)-V(tasa=r-hr))/(2*hr)]


def tipos_tesoro(ahora, carpeta):
    for ano in [ahora.year, ahora.year-1]:
        url = ('https://home.treasury.gov/resource-center/data-chart-center/interest-rates/'
               f'daily-treasury-rates.csv/{ano}/all?type=daily_treasury_bill_rates&field_tdr_date_value={ano}&_format=csv')
        respuesta = requests.get(url, timeout=35)
        respuesta.raise_for_status()
        (carpeta / f'tesoro_original_{ano}.csv').write_text(respuesta.text, encoding='utf-8')
        df = pd.read_csv(StringIO(respuesta.text))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df.Date.dt.date <= ahora.tz_convert('America/New_York').date()].sort_values('Date')
        columnas = [(c, re.search(r'(\d+)\s*Weeks?\s*Bank Discount', c, re.I)) for c in df.columns]
        columnas = [(c, int(m[1])) for c, m in columnas if m]
        if not columnas:
            raise ValueError('No se reconocen las columnas Bank Discount del Tesoro.')
        df = df.dropna(subset=[c for c, _ in columnas])
        if len(df):
            fila = df.iloc[-1]
            w = np.array([w for _, w in columnas])
            b = np.array([float(fila[c])/100 for c, _ in columnas])
            D = 1-b*7*w/360
            if np.any(D <= 0):
                raise ValueError('Descuentos del Tesoro invalidos.')
            curva = pd.DataFrame({'semanas': w, 'T': 7*w/365, 'descuento': b, 'r': -np.log(D)/(7*w/365)}).sort_values('T')
            return curva, str(fila.Date.date()), url
    raise ValueError('No hay tipos del Tesoro anteriores a la captura.')


def filtrar(df, S, q, ahora):
    df = df.copy()
    df['mid'] = (df.bid+df.ask)/2
    edad = (ahora-pd.to_datetime(df.lastTradeDate, utc=True)).dt.total_seconds()/86400
    intrinseco = np.maximum(df.signo*(S-df.strike), 0)
    superior = np.where(df.signo == 1, S*np.exp(max(-q, 0)*df.T_anual),
                        df.strike*np.exp(np.maximum(-df.r, 0)*df.T_anual))
    validas = ((df.bid > 0) & (df.ask >= df.bid) & ((df.ask-df.bid)/df.mid <= .20)
               & (df.openInterest >= 10) & edad.between(0, 7) & ((df.strike/S-1).abs() < .05)
               & (df.currency == 'USD') & (df.contractSize == 'REGULAR')
               & (df.mid >= intrinseco) & (df.mid <= superior))
    df['filtro_individual'] = validas
    elegidas = []
    for fecha, g in df[validas].groupby('vencimiento'):
        parejas = g.pivot(index='strike', columns='signo', values=['bid', 'ask'])
        if not all(s in parejas['bid'].columns for s in [-1, 1]):
            continue
        parejas = parejas.dropna()
        if g.r.iloc[0] >= 0 and q >= 0:
            parejas = parejas[parejas['bid'][1]-parejas['ask'][-1] <= S-parejas.index*np.exp(-g.r.iloc[0]*g.T_anual.iloc[0])]
        strikes = sorted(sorted(parejas.index, key=lambda K: abs(K-S))[:12])
        if len(strikes) < 4:
            continue
        parte = g[g.strike.isin(strikes)].copy()
        parte['muestra'] = parte.strike.map({K: 'calibracion' if j % 2 == 0 else 'evaluacion' for j, K in enumerate(strikes)})
        elegidas.append(parte)
    seleccion = pd.concat(elegidas).sort_values(['vencimiento', 'strike', 'signo']).reset_index(drop=True) if elegidas else pd.DataFrame()
    return seleccion, df


def descargar(ticker, años, carpeta):
    inicio = pd.Timestamp.now(tz='UTC')
    yf.set_tz_cache_location(str(BASE / '.cache_yahoo'))
    activo = yf.Ticker(ticker)
    info = activo.info
    if info.get('currency') != 'USD' or info.get('quoteType') not in ['EQUITY', 'ETF'] or info.get('exchangeTimezoneName') != 'America/New_York':
        raise ValueError('Se necesitan acciones/ETF de EEUU en USD, con opciones americanas estandar.')
    historial = activo.history(start=(inicio-pd.DateOffset(years=años)).date(), auto_adjust=False, actions=True)
    if historial.empty or 'Adj Close' not in historial:
        raise ValueError('Yahoo no devuelve cierres ajustados.')
    pagos = historial.loc[(historial.index >= inicio-pd.Timedelta(days=365)) & (historial.index <= inicio), 'Dividends']
    calendario = mcal.get_calendar('NYSE').schedule(start_date=historial.index[0].date(), end_date=(inicio+pd.Timedelta(days=400)).date())
    cierres = calendario.market_close
    completas = set(calendario.index[cierres < inicio].date)
    historial = historial[[d.date() in completas for d in historial.index]].dropna(subset=['Adj Close'])
    historial.to_csv(carpeta / 'historico.csv')
    R = np.log(historial['Adj Close']).diff().dropna()
    if len(R) < 100 or not np.isfinite(R).all() or R.var() <= 0:
        raise ValueError('Se necesitan al menos 100 retornos validos con varianza positiva.')
    minuto = activo.history(period='5d', interval='1m', auto_adjust=False, prepost=False)
    if minuto.empty:
        raise ValueError('No se pudo obtener un spot con hora; se detiene, sin inventar un precio.')
    minuto.to_csv(carpeta / 'spot_1min.csv')
    S, hora_spot = float(minuto.Close.dropna().iloc[-1]), minuto.Close.dropna().index[-1]
    ahora = pd.Timestamp.now(tz='UTC')
    pagos.to_csv(carpeta / 'dividendos_365_dias.csv')
    q = float(pagos.sum()/S)
    tasas, fecha_tasa, url = tipos_tesoro(ahora, carpeta)
    disponibles = list(activo.options)
    if not disponibles:
        raise ValueError('El ticker no tiene cadena de opciones disponible en Yahoo.')
    expiraciones = sorted(set(min(disponibles, key=lambda d: abs((pd.Timestamp(d).date()-ahora.date()).days-dias)) for dias in [30, 60, 90]))
    originales = []
    for fecha in expiraciones:
        cierre = cierres.loc[cierres.index <= pd.Timestamp(fecha)].iloc[-1]
        T = (cierre-ahora).total_seconds()/(365*86400)
        if not tasas['T'].min() <= T <= tasas['T'].max():
            print('Vencimiento fuera del intervalo interpolable del Tesoro:', fecha)
            continue
        cadena = activo.option_chain(fecha)
        for tabla, signo in [(cadena.calls, 1), (cadena.puts, -1)]:
            originales.append(tabla.assign(signo=signo, vencimiento=fecha, T_anual=T,
                                            r=np.interp(T, tasas['T'], tasas.r), descarga_UTC=str(pd.Timestamp.now(tz='UTC'))))
    if not originales:
        raise ValueError('No hay vencimientos utilizables; no se extrapolan tipos.')
    opciones, auditoria = filtrar(pd.concat(originales, ignore_index=True), S, q, ahora)
    auditoria['seleccion'] = auditoria.contractSymbol.map(opciones.set_index('contractSymbol').muestra) if len(opciones) else 'excluida'
    auditoria.to_csv(carpeta / 'opciones_originales.csv', index=False)
    tasas.to_csv(carpeta / 'tipos.csv', index=False)
    meta = {'ticker': ticker, 'captura_UTC': str(ahora), 'hora_spot': str(hora_spot), 'S': S, 'q': q,
            'tesoro_fecha': fecha_tasa, 'tesoro_url': url, 'años': años, 'retornos': len(R),
            'historico_desde': str(R.index[0]), 'historico_hasta': str(R.index[-1]),
            'vencimientos_solicitados': expiraciones, 'contratos_descargados': len(auditoria),
            'contratos_seleccionados': len(opciones), 'N': N}
    (carpeta / 'captura.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return historial, R, S, q, ahora, cierres[cierres > ahora], opciones, meta


def tabla(df, nombre, carpeta, imprimir=True):
    df.to_csv(carpeta / (nombre+'.csv'), index=False)
    texto = '\n'+nombre.upper()+'\n'+df.to_string(index=False, float_format=lambda x: f'{x:.6g}')+'\n'
    with (carpeta / 'informe.txt').open('a', encoding='utf-8') as f:
        f.write(texto)
    if imprimir:
        print(texto)


def evaluar_tiempo(R, carpeta):
    corte = int(.8*len(R))
    v0 = np.var(R.iloc[:corte], ddof=1)
    resultados, parametros, detalle = [], [], pd.DataFrame({'fecha': R.index[corte:].astype(str)})
    for modelo in MODELOS:
        print('Evaluacion temporal:', modelo, flush=True)
        p = ajustar(R.iloc[:corte], modelo)
        loss = perdidas(p, R.to_numpy(), v0, modelo == MODELOS[2])[corte:]
        parametros.append(dict(modelo=modelo, **dict(zip(['mu', 'phi1', 'phi2', 'omega', 'alpha', 'beta', 'nu'], p))))
        if modelo == MODELOS[2]:
            parametros[-1]['nu'] = np.nan
        resultados.append({'modelo': modelo, 'n_train': corte, 'n_test': len(loss), 'NLL': loss[:, 0].mean(), 'QLIKE': loss[:, 1].mean()})
        for j, metrica in enumerate(['NLL', 'QLIKE']):
            detalle[modelo+'_'+metrica] = loss[:, j]
    rng = np.random.default_rng(123)
    n, ic = len(detalle), []
    bloque = int(np.ceil(n**(1/3)))
    indices = (rng.integers(0, n, (2000, int(np.ceil(n/bloque)), 1))+np.arange(bloque)) % n
    indices = indices.reshape(2000, -1)[:, :n]
    for modelo in MODELOS[1:]:
        for metrica in ['NLL', 'QLIKE']:
            diferencia = (detalle[MODELOS[0]+'_'+metrica]-detalle[modelo+'_'+metrica]).to_numpy()
            bajo, alto = np.quantile(diferencia[indices].mean(axis=1), [.025, .975])
            ic.append({'comparacion': 'AR2-GARCH-t menos '+modelo, 'metrica': metrica,
                       'diferencia': diferencia.mean(), 'IC95_inferior': bajo, 'IC95_superior': alto, 'bloque': bloque})
    for nombre, datos in [('parametros_train', parametros), ('evaluacion_temporal', resultados), ('intervalos_temporales', ic)]:
        tabla(pd.DataFrame(datos), nombre, carpeta)
    tabla(detalle, 'perdidas_temporales', carpeta, False)
    return pd.DataFrame(resultados)


def cotizar(df, S, q, curva, n=N, americana=True):
    precios = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby('vencimiento'):
        precios.loc[g.index] = arbol(S, g.strike.to_numpy(), g.signo.to_numpy(), g.T_anual.iloc[0],
                                     g.r.iloc[0], q, curva, n, americana)
    return precios


def calibrar(df, S, q, t, A, adaptativo):
    def curva(x):
        return (t, A*np.exp(x)) if adaptativo else (t, t*np.exp(2*x))
    def objetivo(x):
        try:
            return np.mean((cotizar(df, S, q, curva(x))-df.mid)**2)
        except (ValueError, FloatingPointError):
            return 1e100
    centro = 0 if adaptativo else np.log(np.sqrt(A[-1]/t[-1]))
    ancho = np.log(4)
    for _ in range(5):
        opt = minimize_scalar(objetivo, bounds=(centro-ancho, centro+ancho), method='bounded', options={'xatol': 1e-6})
        if opt.success and opt.fun < 1e99 and abs(opt.x-centro) < ancho-.02:
            return np.exp(opt.x), opt.fun
        ancho *= 1.6
    raise ValueError('Calibracion sin minimo interior admisible; no se acepta un parametro de borde.')


def volatilidad_iv(fila, S, q, t):
    K, signo, T, r, M = fila.strike, fila.signo, fila.T_anual, fila.r, fila.mid
    f = lambda sigma: arbol(S, K, signo, T, r, q, (t, sigma*sigma*t))[0]-M
    bajo, alto = max(.00001, abs(r-q)*np.sqrt(T/N)*1.001), 1.
    if f(bajo) >= -1e-7:
        return np.nan                 # IV inexistente o no identificable en el intrinseco.
    while f(alto) < 0 and alto < 16:
        alto *= 2
    return brentq(f, bajo, alto, xtol=1e-7) if f(alto) >= 0 else np.nan


def metricas(df):
    filas = []
    for muestra, grupo in df.groupby('muestra'):
        conjuntos = [('total', grupo)] + list(grupo.groupby('vencimiento'))
        for vencimiento, g in conjuntos:
            for modelo in PRECIOS:
                e = g[modelo]-g.mid
                filas.append({'muestra': muestra, 'vencimiento': vencimiento, 'modelo': modelo, 'n': len(g),
                              'MAE': e.abs().mean(), 'RMSE': np.sqrt((e*e).mean()), 'sesgo': e.mean(),
                              'dentro_bid_ask_pct': 100*((g[modelo] >= g.bid) & (g[modelo] <= g.ask)).mean()})
    return pd.DataFrame(filas)


def escenarios(S, T, r, q, curva):
    a, fechas, p = malla(T, r, q, curva, N)
    masa, filas = np.array([1.]), []
    for k in range(N+1):
        nodos = S*np.exp(a*(2*np.arange(k+1)-k))
        cuantiles = nodos[np.minimum(np.searchsorted(masa.cumsum(), [.05, .5, .95]), k)]
        filas.append([fechas[k]*365, *cuantiles, np.sum(masa*nodos)])
        if k < N:
            siguiente = np.zeros(k+2)
            siguiente[:-1] += (1-p[k])*masa
            siguiente[1:] += p[k]*masa
            masa = siguiente
    return pd.DataFrame(filas, columns=['dias', 'q05_Q', 'mediana_Q', 'q95_Q', 'media_Q']), nodos, masa


def graficos(hist, R, p, S, q, t, A, kappa, sigma, df, met, carpeta):
    plt.close('all')
    _, e, h = estados(p, R.to_numpy(), R.var())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].plot(hist.index, hist['Adj Close']); axes[0, 0].set_title('Cierre ajustado historico')
    axes[0, 1].plot(R.index, 100*R, linewidth=.6); axes[0, 1].set_title('Retornos diarios (%)')
    axes[1, 0].plot(R.index, np.sqrt(252*h)*100, label='GARCH')
    axes[1, 0].axhline(R.std()*np.sqrt(252)*100, color='orange', label='Constante historica')
    axes[1, 0].set_title('Volatilidad anualizada (%)'); axes[1, 0].legend()
    axes[1, 1].hist(e[2:]/np.sqrt(h[2:]), bins=40, density=True, label='Residuos estandarizados')
    z, nu = np.linspace(-5, 5, 300), p[-1]
    densidad = np.exp(gammaln((nu+1)/2)-gammaln(nu/2)) / np.sqrt(np.pi*(nu-2)) * (1+z*z/(nu-2))**(-(nu+1)/2)
    axes[1, 1].plot(z, densidad, label='Student estimada'); axes[1, 1].legend()
    fig.suptitle('Historico y estimacion'); fig.savefig(carpeta / '01_historico.png', dpi=180)
    primera = df[df.vencimiento == df.vencimiento.min()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, signo, titulo in [(axes[0, 0], 1, 'Calls'), (axes[0, 1], -1, 'Puts')]:
        g = primera[primera.signo == signo].sort_values('strike')
        ax.fill_between(g.strike, g.bid, g.ask, alpha=.2, label='Bid-ask')
        for col in ['mid', 'Adapt_cal', 'CRR_cal']:
            ax.plot(g.strike, g[col], '.-', label=col)
        ax.set_title(titulo+' '+str(g.vencimiento.iloc[0])); ax.set_xlabel('Strike'); ax.legend(fontsize=8)
    prueba = df[df.muestra == 'evaluacion']
    for col in ['Adapt_cal', 'CRR_cal']:
        axes[0, 2].scatter(prueba.mid, prueba[col], s=15, label=col)
        axes[1, 0].scatter(prueba.strike/S, prueba[col]-prueba.mid, s=15, label=col)
    limites = [prueba.mid.min(), prueba.mid.max()]
    axes[0, 2].plot(limites, limites, 'k--'); axes[0, 2].set_title('Evaluacion: modelo frente a mercado')
    axes[0, 2].set_xlabel('Mid USD'); axes[0, 2].set_ylabel('Modelo USD'); axes[0, 2].legend()
    axes[1, 0].axhline(0, color='black'); axes[1, 0].set_title('Errores reservados (USD)'); axes[1, 0].set_xlabel('K/S'); axes[1, 0].legend()
    resumen = met[(met.muestra == 'evaluacion') & (met.vencimiento == 'total')].set_index('modelo')
    resumen[['MAE', 'RMSE']].plot.bar(ax=axes[1, 1], rot=20, title='Error en contratos reservados (USD)')
    axes[1, 2].scatter(df.BS_equivalente, df.Adapt_europeo-df.BS_equivalente)
    axes[1, 2].axhline(0, color='black'); axes[1, 2].set_title('Arbol europeo menos BS (USD)')
    axes[1, 2].set_xlabel('BS de volatilidad equivalente (USD)')
    fig.savefig(carpeta / '02_comparacion.png', dpi=180)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, griega in zip(axes.flat, GRIEGAS):
        for modelo in ['Adapt', 'CRR', 'IV']:
            g = primera[primera.signo == 1].sort_values('strike')
            ax.plot(g.strike, g[modelo+'_'+griega], '.-', label=modelo)
        ax.set_title(griega); ax.set_xlabel('Strike'); ax.legend(fontsize=8)
    g = primera[primera.signo == 1].sort_values('strike')
    axes[1, 2].plot(g.strike, g.acciones_100, '.-', label='Minimax')
    axes[1, 2].set_title('Acciones por 1 call vendida'); axes[1, 2].set_xlabel('Strike')
    fig.suptitle('Griegas y cobertura: calls '+str(primera.vencimiento.iloc[0]))
    fig.savefig(carpeta / '03_griegas.png', dpi=180)
    fila = df.loc[df.T_anual.idxmax()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, nombre, curva in [(axes[0, 0], 'Adaptativo', (t, kappa*A)), (axes[0, 1], 'CRR', (t, sigma*sigma*t))]:
        esc, nodos, prob = escenarios(S, fila.T_anual, fila.r, q, curva)
        tabla(esc, 'escenarios_'+nombre, carpeta, False)
        tabla(pd.DataFrame({'precio': nodos, 'probabilidad_Q': prob}), 'distribucion_'+nombre, carpeta, False)
        ax.fill_between(esc.dias, esc.q05_Q, esc.q95_Q, alpha=.2, label='Cuantiles Q 5-95%')
        ax.plot(esc.dias, esc.mediana_Q, label='Mediana Q')
        ax.plot(esc.dias, esc.media_Q, '--', label='Media Q')
        ax.set_title(nombre+' calibrado'); ax.set_xlabel('Dias'); ax.set_ylabel('Precio USD'); ax.legend()
        centro = (prob > 1e-5)
        axes[1, 0].plot(nodos[centro], prob[centro], '.-', label=nombre)
    axes[1, 0].set_title('Masa de probabilidad Q al vencimiento'); axes[1, 0].set_xlabel('Precio USD'); axes[1, 0].legend()
    horizonte = t <= fila.T_anual
    axes[1, 1].plot(t[horizonte]*365, (A*kappa)[horizonte], label='Adaptativo')
    axes[1, 1].plot(t[horizonte]*365, (sigma*sigma*t)[horizonte], label='CRR')
    axes[1, 1].set_title('Varianza acumulada calibrada'); axes[1, 1].set_xlabel('Dias'); axes[1, 1].legend()
    fig.suptitle('Escenarios del arbol Q: NO predicciones del precio real | '+str(fila.vencimiento))
    fig.savefig(carpeta / '04_escenarios.png', dpi=180)
    plt.show(block=False)


def ejecutar(ticker, años):
    sello = pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S_%f')
    carpeta = BASE / (re.sub(r'[^A-Z0-9_-]', '_', ticker)+'_'+sello)
    carpeta.mkdir(parents=True)
    
    print('\nDescargando datos. Salida:', carpeta, flush=True)
    hist, R, S, q, ahora, cierres, df, meta = descargar(ticker, años, carpeta)
    tabla(pd.DataFrame([meta]), 'datos_captura', carpeta)
    edad = (ahora-pd.Timestamp(meta['hora_spot'])).total_seconds()/60
    print(f'Spot: {S:.4f} USD | antiguedad {edad:.1f} minutos | tipos de {meta["tesoro_fecha"]}')
    print('La hora de la ultima operacion de cada opcion NO fecha su bid/ask.')
    evaluar_tiempo(R, carpeta)
    p = ajustar(R)
    mu, p1, p2, w, a, b, nu = p
    _, e, h = estados(p, R.to_numpy(), R.var())
    h1 = w+a*e[-1]**2+b*h[-1]
    largo = w/(1-a-b)
    previstas = largo+(a+b)**np.arange(len(cierres))*(h1-largo)
    t = np.r_[0, (cierres-ahora).dt.total_seconds().to_numpy()/(365*86400)]
    A = np.r_[0, previstas.cumsum()]
    sigma_hist = R.std()*np.sqrt(252)
    cuarto = a*a*3*(nu-2)/(nu-4)+2*a*b+b*b if nu > 4 else np.inf
    parametros = dict(zip(['mu', 'phi1', 'phi2', 'omega', 'alpha', 'beta', 'nu'], p))
    parametros.update(persistencia=a+b, varianza_largo=largo, h_siguiente=h1, sigma_CRR_hist=sigma_hist,
                      cuarto_momento_existe=cuarto < 1, condicion_cuarto=cuarto,
                      logL=-perdidas(p, R.to_numpy(), R.var())[2:, 0].sum())
    tabla(pd.DataFrame([parametros]), 'parametros_completos', carpeta)
    tabla(pd.DataFrame({'cierre_UTC': cierres.astype(str), 'h_prevista': previstas, 'A': A[1:]}), 'varianza_prevista', carpeta, False)
    if df.empty:
        raise ValueError('Ningun vencimiento conserva 4 parejas tras los filtros. Datos y evaluacion temporal guardados en '+str(carpeta))
    print('Calibrando opciones con N=500 y N=501...', flush=True)
    calibracion = df[df.muestra == 'calibracion']
    kappa, mse_a = calibrar(calibracion, S, q, t, A, True)
    sigma, mse_c = calibrar(calibracion, S, q, t, A, False)
    tabla(pd.DataFrame([{'kappa': kappa, 'sigma_CRR_cal': sigma, 'MSE_cal_adapt': mse_a, 'MSE_cal_CRR': mse_c,
                         'n_cal': len(calibracion), 'n_eval': sum(df.muestra == 'evaluacion')}]), 'calibracion', carpeta)
    curvas = [(t, A), (t, sigma_hist*sigma_hist*t), (t, kappa*A), (t, sigma*sigma*t)]
    for nombre, curva in zip(PRECIOS, curvas):
        df[nombre] = cotizar(df, S, q, curva)
        df['error_'+nombre] = df[nombre]-df.mid
    df['Adapt_europeo'] = cotizar(df, S, q, curvas[2], americana=False)
    df['BS_equivalente'] = bs(S, df.strike, df.signo, df.T_anual, df.r, q, kappa*np.interp(df.T_anual, t, A))
    df['sigma_equivalente'] = np.sqrt(kappa*np.interp(df.T_anual, t, A)/df.T_anual)
    met = metricas(df)
    tabla(met, 'errores_opciones', carpeta)
    tabla(df, 'precios_opciones', carpeta, False)
    for _, g in df.groupby('vencimiento'):
        for nombre, curva in [('Adapt', curvas[2]), ('CRR', curvas[3])]:
            valores = sensibilidad(S, g.strike.to_numpy(), g.signo.to_numpy(), g.T_anual.iloc[0], g.r.iloc[0], q, curva)
            df.loc[g.index, [nombre+'_'+z for z in GRIEGAS]] = np.array(valores).T
    for i, fila in df.iterrows():
        print(f'Griegas e IV: {i+1}/{len(df)}', end='\r', flush=True)
        iv = volatilidad_iv(fila, S, q, t)
        df.loc[i, 'IV_CRR'] = iv
        valores = sensibilidad(S, fila.strike, fila.signo, fila.T_anual, fila.r, q, (t, iv*iv*t)) if np.isfinite(iv) else [np.nan]*5
        df.loc[i, ['IV_'+g for g in GRIEGAS]] = valores
    df['eta'] = (df.Adapt_delta+df.IV_delta)/2
    df['z'] = (df.Adapt_delta-df.IV_delta).abs()/2
    df['acciones_100'] = 100*df.eta
    df['riesgo_delta_100'] = 100*df.z
    tabla(df, 'opciones_resultados', carpeta, False)
    for griega in GRIEGAS:
        tabla(pd.DataFrame([{'griega': griega, 'modelo': modelo, 'n_con_IV': int(df.IV_CRR.notna().sum()),
                             'MAE_frente_IV': (df[modelo+'_'+griega]-df['IV_'+griega]).abs().mean()} for modelo in ['Adapt', 'CRR']]),
              'comparacion_'+griega, carpeta)
    tabla(df[['contractSymbol', 'muestra', 'bid', 'ask', 'Adapt_cal', 'CRR_cal', 'eta', 'acciones_100', 'riesgo_delta_100']], 'cobertura', carpeta)
    print('Acciones positivas: comprar; negativas: vender, por UN contrato vendido. NaN: IV no identificable, sin cobertura calculada.')
    convergencia = []
    for pasos in [100, 250, 500, 1000]:
        for nombre, curva in [('Adapt', curvas[2]), ('CRR', curvas[3])]:
            precios = cotizar(df, S, q, curva, n=pasos)
            error = precios[df.muestra == 'evaluacion']-df.loc[df.muestra == 'evaluacion', 'mid']
            convergencia.append({'N': pasos, 'modelo': nombre, 'MAE_eval': error.abs().mean(),
                                 'max_diferencia_precio_vs_N500': (precios-df[nombre+'_cal']).abs().max()})
    tabla(pd.DataFrame(convergencia), 'convergencia_sin_recalibrar', carpeta)
    graficos(hist, R, p, S, q, t, A, kappa, sigma, df, met, carpeta)
    print('\nTablas CSV, informe TXT y 4 figuras PNG guardados en:', carpeta)
    return carpeta


def main():
    print('ARBOL ADAPTATIVO / CRR - Articulo de Jose Maria Lopez Espinosa')
    print('Solo EEUU/USD con opciones estandar. Yahoo puede dar datos retrasados.')
    ticker = (input('Ticker [SPY]: ').strip() or 'SPY').upper()
    años = int(input('años de historico [5]: ').strip() or 5)
    segundos = int(input('Actualizar cada cuantos segundos (0 = al cerrar los graficos) [0]: ').strip() or 0)
    if años < 1 or segundos < 0:
        raise ValueError('años >= 1; segundos >= 0.')
    while True:
        try:
            ejecutar(ticker, años)
        except Exception as error:
            print('\nNo se completa esta captura:', error)
            print('No se sustituyen datos ni parametros por valores inventados.')
            if input('Enter para reintentar; q para salir: ').strip().lower() == 'q':
                break
            continue
        if segundos:
            print(f'Proxima captura en {segundos} segundos tras el calculo. Ctrl+C para salir.')
            fin = time.monotonic()+segundos
            while time.monotonic() < fin:
                plt.pause(.2)
        else:
            print('Cierra las ventanas para volver a la consola.')
            plt.show()
            if input('Enter actualiza y recalibra; q sale: ').strip().lower() == 'q':
                break


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nFin.')
