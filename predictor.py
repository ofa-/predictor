#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

import locale; locale.setlocale(locale.LC_ALL, "fr_FR.UTF8")
import numpy as np

from matplotlib.dates import DateFormatter
from matplotlib import patheffects

from sklearn.linear_model import LinearRegression


regions = {
        "pc":  [ "Petite Couronne", "92|93|94" ],
        "gc":  [ "Grande Couronne", "77|78|91|95" ],
        "idf": [ "Île de France", "75|91|92|93|94|95|77|78" ],
        "sud": [ "13|30|34|83" ],
        "ra":  [ "69|38|01|26" ],
        "brg": [ "Bourgogne Franche-Comté", "21|25|39|58|70|71|89|90" ],
        "fc":  [ "Franche-Comté", "25|39|70|90" ],
        "met": [ "Métropole", "met" ],
}

def main():

    data = pd.read_csv('data.csv', sep=";", parse_dates=['jour'])

    metropole = data[~data.dep.str.match("^97")]

    arg = "|".join(opt.arg)

    region = regions[arg][-1] if arg in regions else arg

    selection = metropole if arg == "met" else \
                data[data.dep.str.match(region)]

    sums = selection.groupby('jour').sum()

    data = sums.rolling(7).mean()

    dc_ref, dc_noise = avg_dc_line(region)
    reg_line, chunks = regressor(data)
    pred, cuts = predictor(data)

    incid = data[["incid_rea", "incid_dc"]]
    if opt.round:
        incid = incid.round()

    incid = incid \
            .join(reg_line) \
            .join(pred, how='outer')

    if opt.pred: pass
    elif opt.nopred or arg != "met":
        incid['pred'] = None
        pred = cuts = []

    with plt.style.context(opt.style) if opt.style else plt.xkcd():
        plot = incid.plot(logy=True)
        show_dbl(plot, reg_line, chunks)
        annotate(plot, pred, cuts)

        avg_dc_percent = 50
        avg_dc = plot_avg_dc(plot, dc_ref, avg_dc_percent)
        if opt.round:
            dc_noise = dc_noise.round()
        dc_noise.plot(linestyle=":", linewidth=.5, color="grey", zorder=0) \
                    if opt.noise else None

        if opt.fouché:
            (data.incid_rea * 5/8).rename('Fouché-fix réa') \
                    .plot(linestyle="--", linewidth=.7, color="#00D")

        # light => alpha area: .05, markers: .25
        fill(sums.incid_dc,     alpha=.1, color="orange", zorder=-1)
        sums.incid_dc.plot     (alpha=.3, color="orange", zorder=-1,
                                marker="+", linestyle="")

        set_opts(plot)
        set_view(plot, arg, gap = cuts[-1][1] if cuts else 0)
        set_title(plot, arg, double_times(data, chunks[-2:]))

        x = pd.Timestamp(plot.axes.get_xlim()[0], unit="D")
        add_note(plot, x, avg_dc, f"{avg_dc_percent}%")
        add_note(plot, x, dc_noise, f"bruit") \
                if opt.noise else None

        plot.figure.savefig(arg + ("-full" if opt.full else ""))


def fill(line, **kwargs):
        line.plot(linestyle="-", **kwargs) \
        .axes.fill_between(line.index, line, 0.1, **kwargs)


def add_note(plot, x, data, text, side=True):
    plot.annotate(text, color="#AAA", size="x-small",
                    xy=(x+pd.Timedelta(days=1), data[x]))


def plot_avg_dc(plot, dc_ref, dc_percent):
        avg_dc = dc_ref * dc_percent
        if opt.round:
            avg_dc = avg_dc.round()
        avg_dc.plot(linestyle=":", linewidth=.5, color="grey", zorder=0)
        return avg_dc


def regressor(data):
    reg_data = data['incid_rea']

    chunks = [
                [9,14],
                [19,28],
                [30,100],
                [120,140],
                [150,160],
                [172,192],
                [193,206],
                [207,215],
                [216,229],
                [230,237],
                [240,len(data)],
            ]

    reg_line = pd.concat([
        exp_lin_reg(reg_data[range(*chunk)])
            for chunk in chunks ])

    return reg_line, chunks


def predictor(data):
    scaled = 10**0.2 * data['incid_rea'].pow(0.885)

    shifts = [
                [ [0,20], 5 ],
                [ [19,26], 13 ],
                [ [25,40], 15 ],
                [ [40,66], 17 ],
                [ [70,91], 14 ],
                [ [84,136], 27 ],
                [ [135,144], 32 ],
                [ [153,159], 23 ],
                [ [165,172], 17 ],
                [ [172,179], 19 ],
                [ [178,185], 26 ],
                [ [207,213], 8 ],
                [ [219,225], 4 ],
                [ [230,231], 0 ],
                #[ [228,len(scaled)], 1 ],
            ]

    shifted = pd.concat([
        scaled[range(*chunk[0])] .shift(periods=chunk[1], freq='D')
            for chunk in shifts
        ]) \
        .rename('pred')

    return shifted, shifts


def avg_dc_line(region):
    dc_j = pd.read_csv("dc_j.csv")

    sel = dc_j.depdom.str.match(region) if region != "met" else \
         ~dc_j.depdom.str.match("9[7-9]|na")

    dc = dc_j[sel].groupby(['ADEC', 'MDEC', 'JDEC']).dc_j.sum()
    avg = dc.groupby('MDEC').mean()
    std = dc.groupby('MDEC').std()

    dates = pd.date_range("2020-03-01", "2021-04-01")
    avg_dc = [ avg[month]/100 for month in dates.month ]
    noise =  [ std[month] for month in dates.month ]

    return pd.Series(index=dates, data=avg_dc).rename('dc_j'), \
           pd.Series(index=dates, data=noise).rename('dc_noise')


def double_times(data, chunks):
    data = data['incid_rea']
    return [ double_time(data[range(*chunk)]) for chunk in chunks ]


def double_time(data):
    # y = ax + b  =>  dy = a dx  =>  dx = 1/a dy
    # y = log(u)  =>  dy = log(u2) - log(u1) = log(u2/u1)
    # u2/u1 = 2   =>  dx = 1/a log(2)

    return np.log(2) / slope(data)


def annotate(plot, shifted, shifts):
    curr = 0
    for shift in shifts:
        chunk, nb_days = shift
        size = len(range(*chunk))
        middle = int(size/2)
        print(chunk, size, nb_days)
        side = shifted[middle+curr] - shifted[curr] < 0
        point = (
            shifted.index[middle+curr],
            shifted[middle+curr]
        )
        _annotate(plot, point, nb_days, side) if size > 1 else None
        curr += size


def _annotate(plot, point, nb_days, side):
    plot.annotate(
        f'+{nb_days} j ',
        xy=point,
        bbox=dict(boxstyle="round4", fc="w"),
        arrowprops=dict(arrowstyle="-|>",
            connectionstyle="arc3,rad="+("" if side else "-")+"0.2", fc="w"),
        xytext=(
            point[0] + pd.Timedelta(days=3),
            point[1] * (1.2 if side else 1/1.2) # logscale => offset via * or /
        )
    )


def show_dbl(plot, reg_line, chunks):
    size = [ len(range(*chunk)) for chunk in chunks ]
    spots = [ int(size[i]/2) + sum(size[:i]) for i in range(len(size)) ]
    for spot in spots:
        nb_days = double_time(reg_line[spot-1:spot+1])
        point = (reg_line.index[spot], reg_line[spot])
        if abs(nb_days) > 90:
            continue
        plot.annotate(
            f'{abs(round(nb_days))}',
            fontsize="x-small",
            color="green",
            bbox=dict(boxstyle="circle", color="green", alpha=.2),
            xy=text_xy(point, nb_days),
            path_effects=[
                patheffects.withStroke(linewidth=4, foreground="w"),
            ]
        )


def text_xy(point, nb_days):
    a = np.log(2)/nb_days
    d = 1.1 if nb_days > 0 else 1.2
    dy = d/np.sqrt(1 + a**2)
    dx = d/np.sqrt(1 + 1/a**2)
    return (
        point[0] - pd.Timedelta(hours=dx),
        point[1] * (dy if nb_days > 0 else 1/dy)
    )


def set_opts(plot):
    log_scalator = [1,2,3,4,5,7,10,20,30,50,70,100,200,300,500,900,1500]
    int_formatter = lambda x, pos: f'{x:.0f}'
    plot.axes.yaxis.set_minor_locator(plt.FixedLocator(log_scalator))
    plot.axes.yaxis.set_major_locator(plt.FixedLocator(log_scalator))
    plot.axes.yaxis.set_minor_formatter(int_formatter)
    plot.axes.yaxis.set_major_formatter(int_formatter)
    plot.axes.xaxis.set_major_formatter(DateFormatter('\xAF\n%b'))
    plot.axes.set_xlabel(None)
    plot.axes.tick_params(which='both', right=True, labelright=True)
    plot.axes.tick_params(which='both', axis="y", length=6, width=1)
    plot.axes.set_xticks(plot.axes.get_xticks()[:-1])

    plot.legend(["admis. réa / j", "nouv. dc / j"], fontsize="small")
    plot.figure.set(figwidth=6, figheight=6)
    plot.figure.subplots_adjust(bottom=.12, left=.11, right=.89, top=.85)


def set_view(plot, arg, gap):
    td = pd.Timedelta
    now = pd.to_datetime("now")
    date = pd.to_datetime
    gap = 2 if gap < 2 else gap

    if opt.episode_1:
        now = date("2020-03-22") + td(days=64 if opt.two_months else 35)

    plot.set(xlim=(now-td(days=33), now+td(days=2)))
    zoom_1_50_adaptive(plot, arg)


    if arg == "met": # add room for predictor, keep xscale 35d
        plot.set( xlim=(now-td(days=35-gap), now+td(days=gap)))

    if opt.two_months:
        last_2_months = (now-td(days=62), now+td(days=2))
        plot.figure.set(figwidth=8, figheight=6)
        plot.set(xlim=last_2_months)

    if opt.zoom_1_100:
        zoom_1_100(plot, arg)

    if opt.full:
        fig_xsize = 16 * ( (now - date("2020-03-20")).days /
            (date("2020-10-20") - date("2020-03-20")).days )
        plot.figure.set(figwidth=fig_xsize, figheight=6)
        plot.set(xlim=("2020-03-20", now+td(days=gap)))
        zoom_full_adaptive(plot, arg)


def zoom_full_adaptive(plot, arg):
    yscale = pd.Series([0.8, 64])

    factor = 30 if arg == "met" else \
              7 if arg == "idf" else \
              4 if arg in [ "pc", "gc" ] else \
              1

    plot.set(ylim=(yscale * factor).values)


def zoom_1_10_adaptive(plot, arg):
    yscale = pd.Series([0.4, 16])

    factor = 21 if arg == "met" else \
              8 if arg == "idf" else \
              4 if arg in [ "gc", "pc" ] else \
              1

    plot.set(ylim=(yscale * factor).values)


def zoom_1_50_adaptive(plot, arg):
    yscale = pd.Series([0.8, 64])

    factor = 30 if arg == "met" else \
              4 if arg == "idf" else \
              2 if arg in [ "pc", "gc" ] else \
              1

    plot.set(ylim=(yscale * factor).values)


def zoom_1_100(plot, arg):
    yscale = pd.Series([0.8, 128])

    factor = 7.15 if arg == "met" else \
             1

    plot.set(ylim=(yscale * factor).values)


def exp_lin_reg(reg_data):
    line, slope = _exp_lin_reg(reg_data)
    return line

def slope(reg_data):
    line, slope = _exp_lin_reg(reg_data)
    return slope


def _exp_lin_reg(reg_data):
    X = reg_data.index.values.reshape(-1,1)
    Y = reg_data.mask(reg_data == 0).apply(np.log)

    reg = LinearRegression()
    reg.fit(X, Y.fillna(0))

    reg_line = reg.predict(X.astype('float64'))
    slope = reg_line[1]-reg_line[0]

    return pd.Series(index=Y.index, data=reg_line) \
                .rename('reg') \
                .apply(np.exp) \
            , slope


def pop_info(arg):
    region = regions[arg][-1] if arg in regions else arg

    metropole = dep[~dep.NUMÉRO.str.match("97")].POPULATION.sum()
    pop_region = dep[dep.NUMÉRO.str.match(region)].POPULATION.sum()

    return pop_region / metropole


def pop_info_string(arg):
    return "" if arg == "met" else \
            "({:.0f}% de la population)".format(pop_info(arg)*100)


def set_title(plot, arg, dbl_time):
    if arg in regions:
        region = regions[arg][0]
    elif "|" in arg:
        region = arg
    else:
        region = dep[dep.NUMÉRO == arg].NOM.values[0]

    region = region.replace("|", " ")

    pop_info = pop_info_string(arg)
    rea_x2_time = rea_time_change_string(dbl_time)

    title = f"{region} {pop_info}"
    title += f"\n{rea_x2_time}" \
                if not opt.episode_1 else "\n(Épisode 1)"

    plot.set_title(title, pad=20, fontsize="small")


def rea_time_change_string(dbl_time):
    prev = pretty_time(dbl_time[0])
    curr = pretty_time(dbl_time[1])

    if prev[0] == "-":
        head = f"/2 : {prev[1:]}"
        tail = f"{curr[1:]}"   if curr[0] == "-" else \
               f"x2 en {curr}" if curr != "_" else curr
    else:
        head = f"x2 : {prev}"
        tail = f"/2 en {curr[1:]}" if curr[0] == "-" else \
               f"{curr}"

    return f"réa {head} –> {tail}"


def pretty_time(val):
    return f"_" if abs(val) > 90 else f"{val:.0f} j"


def init():
    from os import path, system
    if not path.exists("data.csv"):
        system("fetch.sh")

    global dep
    dep = pd.read_csv("dep.csv", sep="\t")

    global opt
    opt = parse_args()


def show():
    from os import getenv
    if getenv("DISPLAY") and not opt.noshow: plt.show()


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--zoom-1-100", action="store_true",
            help="graph using 1-100 y scale [default is 1-50]")
    parser.add_argument("--two-months", action="store_true",
            help="graph last two months")
    parser.add_argument("--full", action="store_true",
            help="graph full history from day one")
    parser.add_argument("--episode-1", action="store_true",
            help="graph Episode I time window")
    parser.add_argument("--fouché", action="store_true",
            help="graph Fouché-fixed réa (5/8)")
    parser.add_argument("--pred", action="store_true",
            help="graph predictor anyway [normaly only for met]")
    parser.add_argument("--nopred", action="store_true",
            help="don't show predictor graph")
    parser.add_argument("--noise", action="store_true",
            help="show mortality noise level")
    parser.add_argument("--round", action="store_true",
            help="show rounded values graphs")
    parser.add_argument("--style", action="store",
            choices=plt.style.available, metavar='<style>',
            help="use <style> instead of xkcd [try: fast]")
    parser.add_argument("--noshow", action="store_true",
            help="don't display graph on screen")
    parser.add_argument('arg', nargs='+',
            help="dept [dept ...] or region (%s)" % "|".join(regions.keys()))

    return parser.parse_args()


init()
main()
show()
