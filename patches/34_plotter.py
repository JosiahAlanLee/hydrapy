"""Line work for a pen plotter, from the same fields as everything else.

Run it as a script to write three SVGs (millimetres, A4 landscape, paths ordered to cut pen-up
travel; vpype and the AxiDraw software both read them):

    python patches/34_plotter.py

    plot_contours.svg      isolines of a noise field (topography)
    plot_flow.svg          streamlines through the same field read as angles (the flow-field plot)
    plot_harmonograph.svg  a damped double pendulum, the original plotter

As a patch (hydrapy live / render), o0 shows the field and o1 the contours drawn back as a picture.
"""
from hydrapy import *

field = noise(1.4, 0.0).add(osc(2, 0, 0).rotate(0.6), 0.25).contrast(1.2)
field.out(o0)
field.contours(9, 0.08, 0.0).out(o1)

if __name__ == "__main__":
    to_plotter_svg(chain_contours(field, levels=9, size=(420, 300)), "plot_contours.svg")
    to_plotter_svg(streamlines(field, n=900, steps=120, step=0.008, size=(300, 210)), "plot_flow.svg")
    to_plotter_svg(Harmonograph(2, 3, 2.004, 3.003, 0.3, 0.04, 40, 0.95, 0, points=16000).at(0), "plot_harmonograph.svg")
    print("wrote plot_contours.svg, plot_flow.svg, plot_harmonograph.svg")

render(o0)
