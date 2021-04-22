#!/usr/bin/env python
# coding: utf-8


# Copyright (C) 2021 Yoshinta Setyawati <yoshintaes@gmail.com>

# Visualization of SXS and analytic waveform model
# Use by executing: bokeh serve main.py
# command to run at your command prompt.
# Then navigate to the URL http://localhost:5006/main in your browser.


import numpy as np
import os
import h5py
import json
import glob
from pycbc.waveform import get_fd_waveform,amplitude_from_frequencyseries,phase_from_frequencyseries,fd_approximants, get_td_waveform, td_approximants
from pycbc import types
from bokeh.models import TableColumn,ColumnDataSource,DataTable
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models.widgets import  Select, PreText,Panel, Tabs, Slider
from bokeh.io import curdoc
from bokeh.layouts import column, row, grid, layout
from scipy.interpolate import InterpolatedUnivariateSpline as unispline

# =============================================================================
#                                   First panel
# =============================================================================
def open_sxs(sxs_data):
    #https://data.black-holes.org/waveforms/documentation.html
    #AhA=apparent horizon A
    #AhB=apparent horizon B
    #AhC=common apparent horizon
    with h5py.File(  os.path.join(sxs_data, "Horizons.h5" ), 'r') as f:
        AhA = f['AhA.dir/CoordCenterInertial.dat'][:]
        AhB = f['AhB.dir/CoordCenterInertial.dat'][:]
        AhC = f['AhC.dir/CoordCenterInertial.dat'][:]
    return AhA,AhB,AhC


def sep_time(horizon1, horizon2):
    hor_times = horizon1[:,0]
    dx=horizon1[:,1]-horizon2[:,1]
    dy=horizon1[:,2] - horizon2[:,2]
    sep_xy = np.array([horizon1[:,1]-horizon2[:,1], horizon1[:,2] - horizon2[:,2]])
    sep = np.sqrt( sep_xy[0,:]**2. + sep_xy[1,:]**2. )
    return hor_times,sep,dx[:2000],dy[:2000]

def moving_average(t, x, seglen, nseg):
    dt = t[1] - t[0]
    means = []
    times = []
    st = 0
    for i in range(int(len(x)/nseg)):
        en = int(st + seglen/dt)
        try:
            times.append(t[st])
            means.append(np.mean(x[st:en]))
        except:
            break
        st = st + seglen
    return times, means

def get_h22(sxs_data):
    with h5py.File(  os.path.join(sxs_data, "rhOverM_Asymptotic_GeometricUnits_CoM.h5" ), 'r') as f:
        h22 = f["OutermostExtraction.dir/Y_l2_m2.dat"][:]
        h2m2 = f["OutermostExtraction.dir/Y_l2_m-2.dat"][:]
    times = h22[:,0]
    for t1, t2 in zip(times, h2m2[:,0]):
        assert t1 == t2
    h22 = h22[:,1] + 1.j * h22[:,2]
    h2m2 = h2m2[:,1] + 1.j * h2m2[:,2]
    return times,h22, h2m2

def get_hlm(sxs_data):
    with h5py.File(  os.path.join(sxs_data, "rhOverM_Asymptotic_GeometricUnits_CoM.h5" ), 'r') as f:
        h21 = f["OutermostExtraction.dir/Y_l2_m1.dat"][:]
        h2m1 = f["OutermostExtraction.dir/Y_l2_m-1.dat"][:]
        h20 = f["OutermostExtraction.dir/Y_l2_m0.dat"][:]
        h22 = f["OutermostExtraction.dir/Y_l2_m2.dat"][:]
        h2m2 = f["OutermostExtraction.dir/Y_l2_m-2.dat"][:]
    times = h22[:,0]
    for t1, t2 in zip(times, h2m2[:,0]):
        assert t1 == t2
    h21 = h21[:,1] + 1.j * h21[:,2]
    h2m1 = h2m1[:,1] + 1.j * h2m1[:,2]
    h20 = h20[:,1] + 1.j * h20[:,2]
    h2m2 = h2m2[:,1] + 1.j * h2m2[:,2]
    h22 = h22[:,1] + 1.j * h22[:,2]
    norm_time=times-times[np.argmax(h22)]
    return norm_time,h21, h2m1,h2m2, h22,h20

def get_data(which_data):
    AhA,AhB,AhC=open_sxs(which_data)
    hor_times,sep,dx,dy=sep_time(AhA,AhB)
    mov_avg_sep_t, mov_avg_sep = moving_average(hor_times, sep, seglen=1000, nseg=10)
    times,h22,h2m2=get_h22(which_data)
    phase22 = np.unwrap(np.angle(h22))
    freq22 = unispline(times, phase22).derivative()(times)
    norm_time=times-times[np.argmax(h22)]
    return AhA, AhB, norm_time, h22,phase22,freq22, hor_times, sep, mov_avg_sep_t,mov_avg_sep,dx,dy

# =============================================================================
#                                   Second panel
# =============================================================================

def q_to_masses(mass_rat,total_m):
    mass1=mass_rat/(mass_rat+1)*total_m
    mass2=total_m-mass1
    return mass1,mass2

def generate_analytic_waveform(mass_rat, eccentricity,approximant='TaylorF2Ecc',total_m=50,f_lower=20.,delta_f=0.1):
    mass1,mass2=q_to_masses(mass_rat,total_m)
    hp,hc=hp,hc=get_fd_waveform(mass1=mass1,mass2=mass2,delta_f=delta_f,f_lower=f_lower, approximant=approximant,eccentricity=eccentricity)
    hs=hp+hc*1j
    amp=amplitude_from_frequencyseries(types.FrequencySeries(hs,delta_f=delta_f))
    phase=phase_from_frequencyseries(types.FrequencySeries(hs,delta_f=delta_f))
    return hp.sample_frequencies.data,np.real(hp.data),np.real(hc.data),np.imag(hp.data),np.imag(hc.data),amp.data,phase.data

freq,hp_real,hc_real,hp_imag,hc_imag,amp,phase=generate_analytic_waveform(mass_rat=1.,eccentricity=0)
dic_p2 = {'hp_real':hp_real,'hc_real':hc_real,'hp_imag':hp_imag,'hc_imag':hc_imag,'amp':amp,'phase':phase,'freq':freq}
sourcep2=ColumnDataSource(data=dic_p2)

pn21=figure(title='h+',x_axis_label='freq(Hz)',y_axis_label='strain', plot_width=500, plot_height=400)
n11=pn21.line(x='freq', y='hp_real',source = sourcep2, color='blue',legend='Re{h+}')
n12=pn21.line(x='freq', y='hp_imag',source = sourcep2, color='orange',legend='Im{h+}')
pn21.toolbar.logo = None
pn21.legend.click_policy="hide"
lines=[n11,n12]

pn22=figure(title='hx',x_axis_label='freq(Hz)',y_axis_label='strain', plot_width=500, plot_height=400)
n21=pn22.line(x='freq', y='hc_real',source = sourcep2, color='blue',legend='Re{hx}')
n22=pn22.line(x='freq', y='hc_imag', source = sourcep2, color='orange',legend='Im{hx}')
pn22.toolbar.logo = None
pn22.legend.click_policy="hide"
lines=[n21,n22]


pn23=figure(title='Amplitude',x_axis_label='freq(Hz)',y_axis_label='strain', plot_width=500, plot_height=400)
pn23.line(x='freq', y='amp',source = sourcep2, color='green',line_width=3)
pn23.toolbar.logo = None

pn24=figure(title='Phase',x_axis_label='freq(Hz)',y_axis_label='rad', plot_width=500, plot_height=400)
pn24.line(x='freq', y='phase',source = sourcep2, color='red',line_width=3)
pn24.toolbar.logo = None


q_slider = Slider(start=1, end=10, value=1, step=.5, title="Mass ratio (q)")
e_slider = Slider(start=0., end=0.9, value=0, step=.05, title="Eccentricity (e)")
model_select = Select(title="FD Models",  options=fd_approximants())

def update_slider(attrname, old, new):
    # Get the current slider values
    q = q_slider.value
    e = e_slider.value
    approximant = model_select.value
    freq,hp_real,hc_real,hp_imag,hc_imag,amp,phase=generate_analytic_waveform(mass_rat=q,eccentricity=e,approximant=approximant)
    sourcep2.data = {'hp_real':hp_real,'hc_real':hc_real,'hp_imag':hp_imag,'hc_imag':hc_imag,'amp':amp,'phase':phase,'freq':freq}

for w in [q_slider,e_slider,model_select]:
    w.on_change('value', update_slider)

layoutan=row(column(pn21,pn22),column(pn23,pn24),column(q_slider,e_slider,model_select))

# =============================================================================
#                                   Third panel
# =============================================================================

def update_table2(attrname, old, new):
    try:
        selected_index = source31.selected.indices[0]
        sval=source31.data["Simulation"][selected_index]
        which_data = data_path+sval  # changed this to the dict
        time_hlm,h21, h2m1,h2m2lm, h22lm,h20=get_hlm(which_data)
        source32.data={'time_hlm':time_hlm,'h21real':h21.real,'h21imag':h21.imag, 'h2m1real':h2m1.real,'h2m1imag':h2m1.imag,'h2m2lmreal': h2m2lm.real,'h2m2lmimag':h2m2lm.imag, 'h22lmreal': h22lm.real,'h22lmimag':h22lm.imag,'h20real':h20.real,'h20imag':h20.imag}
    except IndexError:
        pass


# =============================================================================
#                                   Fourth panel
# =============================================================================


def generate_TD_waveform(mass_rat, eccentricity,s1z,s2z,approximant='TaylorT1',total_m=50,f_lower=20.,delta_t=1./1024):
    nonspinning_models=['TaylorT1','TaylorT2','TaylorEt','TaylorT3','TaylorT4','EOBNRv2','EOBNRv2HM','EOBNRv2_ROM','EOBNRv2HM_ROM','SEOBNRv1','SEOBNRv1_ROM_DoubleSpin','SEOBNRv1_ROM_EffectiveSpin','TEOBResum_ROM','PhenSpinTaylor','PhenSpinTaylorRD','IMRPhenomA','EccentricTD','NRSur7dq2']
    if approximant in nonspinning_models:
        s1z=0
        s2z=0
    mass1,mass2=q_to_masses(mass_rat,total_m)
    hp,hc=hp,hc=get_td_waveform(mass1=mass1,mass2=mass2,spin1z=s1z,spin2z=s2z,delta_t=delta_t,f_lower=f_lower, approximant=approximant,eccentricity=eccentricity)
    hs=hp+hc*1j
    amp=abs(hs)
    phase=np.unwrap(np.angle(hs))
    return hp.sample_times.data,np.real(hp.data),np.real(hc.data),np.imag(hp.data),np.imag(hc.data),amp,phase

timeTD,hp_realTD,hc_realTD,hp_imagTD,hc_imagTD,ampTD,phaseTD=generate_TD_waveform(mass_rat=1.,eccentricity=0,s1z=0.,s2z=0.)
dic_p42 = {'hp_realTD':hp_realTD,'hc_realTD':hc_realTD,'hp_imagTD':hp_imagTD,'hc_imagTD':hc_imagTD,'ampTD':ampTD,'phaseTD':phaseTD,'timeTD':timeTD}
sourcep42=ColumnDataSource(data=dic_p42)

pn41=figure(title='h+',x_axis_label='time(sec)',y_axis_label='strain', plot_width=500, plot_height=400)
n41=pn41.line(x='timeTD', y='hp_realTD',source = sourcep42, color='blue',legend='Re{h+}')
n42=pn41.line(x='timeTD', y='hp_imagTD',source = sourcep42, color='orange',legend='Im{h+}')
pn41.toolbar.logo = None
pn41.legend.click_policy="hide"
lines=[n41,n42]

pn42=figure(title='hx',x_axis_label='time(sec)',y_axis_label='strain', plot_width=500, plot_height=400)
n41=pn42.line(x='timeTD', y='hc_realTD',source = sourcep42, color='blue',legend='Re{hx}')
n42=pn42.line(x='timeTD', y='hc_imagTD', source = sourcep42, color='orange',legend='Im{hx}')
pn42.toolbar.logo = None
pn42.legend.click_policy="hide"
lines=[n41,n42]


pn43=figure(title='Amplitude',x_axis_label='time(sec)',y_axis_label='strain', plot_width=500, plot_height=400)
pn43.line(x='timeTD', y='ampTD',source = sourcep42, color='green',line_width=3)
pn43.toolbar.logo = None

pn44=figure(title='Phase',x_axis_label='time(sec)',y_axis_label='rad', plot_width=500, plot_height=400)
pn44.line(x='timeTD', y='phaseTD',source = sourcep42, color='red',line_width=3)
pn44.toolbar.logo = None


q_slider = Slider(start=1, end=10, value=1, step=.5, title="Mass ratio (q)")
e_slider = Slider(start=0., end=0.9, value=0, step=.05, title="Eccentricity (e)")
s1z_slider = Slider(start=-1, end=1, value=0, step=.05, title="Spin1z")
s2z_slider = Slider(start=-1, end=1, value=0, step=.05, title="Spin2z")
model_select = Select(title="TD Models",  options=td_approximants())

def update_slider2(attrname, old, new):
    # Get the current slider values
    q = q_slider.value
    e = e_slider.value
    s1z = s1z_slider.value 
    s2z = s2z_slider.value
    approximant = model_select.value
    timeTD,hp_realTD,hc_realTD,hp_imagTD,hc_imagTD,ampTD,phaseTD=generate_TD_waveform(mass_rat=q,eccentricity=e,s1z=s1z,s2z=s2z,approximant=approximant)
    sourcep42.data = {'hp_realTD':hp_realTD,'hc_realTD':hc_realTD,'hp_imagTD':hp_imagTD,'hc_imagTD':hc_imagTD,'ampTD':ampTD,'phaseTD':phaseTD,'timeTD':timeTD}

for w in [q_slider,e_slider,s1z_slider,s2z_slider,model_select]:
    w.on_change('value', update_slider2)

layoutTD=row(column(pn41,pn42),column(pn43,pn44),column(q_slider,e_slider,s1z_slider,s2z_slider,model_select))


#tab1 = Panel(child=layoutNR, title="NR data")
tab2 = Panel(child=layoutan,title="Analytic FD")
#tab3 = Panel(child=layout3,title="NR l=2")
tab4 = Panel(child=layoutTD,title="Analytic TD")
#tabs = Tabs(tabs=[tab1,tab3,tab2,tab4],sizing_mode='scale_width')
tabs = Tabs(tabs=[tab2,tab4],sizing_mode='scale_width')
#layout = row(column(p,data_table),column(k,s),r)
curdoc().add_root(tabs)
curdoc().title = "Eccentric Waveforms Visualization"
