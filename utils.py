"""
Utility code for miscellaneous tasks
Created by Matthew Keaton on 4/16/2020
"""


def elapsed_time(seconds, short=False):
    if not short:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1 hour, '
            else:
                e_time += '{:d} hours, '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1 minute, '
            else:
                e_time += '{:d} minutes, '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1 second.'
        else:
            e_time += '{:.1f} seconds.'.format(seconds)
        return e_time
    else:
        minutes = int(seconds // 60)
        hours = int(minutes // 60)
        e_time = ''
        if hours > 0:
            if hours == 1:
                e_time += '1hr '
            else:
                e_time += '{:d}hrs '.format(hours)
            minutes %= 60
            seconds %= 60
        if minutes > 0:
            if minutes == 1:
                e_time += '1m '
            else:
                e_time += '{:d}m '.format(minutes)
            seconds %= 60
        if seconds == 1:
            e_time += '1s'
        else:
            e_time += '{:.1f}s'.format(seconds)
        return e_time
