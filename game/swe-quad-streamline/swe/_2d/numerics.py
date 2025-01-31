
def lax_friedrichs(Ql, Qr, Fl, Fr, max_speed):
    dissipation = max_speed * (Qr - Ql)
    return 0.5 * (Fl + Fr - dissipation)

