#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2010-2019 Sebastian Pipping <sebastian@pipping.org>
# Copyright (C) 2011      Wouter Haffmans <wouter@boxplosive.nl>
# Copyright (C) 2019      Kevin Lane <kevin.lane84@outlook.com>
# Licensed under GPL v3 or later
#
from __future__ import print_function

import datetime
import getpass
import math
import os
import shlex
import signal
import sys
import time
from collections import namedtuple

import pysvn
import six
from six.moves import xrange
from six.moves.urllib.parse import quote as urllib_parse_quote
from six.moves.urllib.parse import urljoin as urllib_parse_urljoin

try:
    import argparse
except ImportError:
    print("ERROR: You need Python 2.7+ unless you have module argparse "
          "(package dev-python/argparse on Gentoo) installed independently.",
          file=sys.stderr)
    sys.exit(1)


_EPILOG = """\
Please report bugs at https://github.com/hartwork/svneverever.  Thank you!
"""


_OsTerminalSize = namedtuple('_OsTerminalSize', ['columns', 'lines'])


def _os_get_terminal_size_pre_3_3(fd=0):
    import fcntl
    import struct
    import termios

    lines, columns, _ph, _pw = struct.unpack('HHHH', (
        fcntl.ioctl(fd, termios.TIOCGWINSZ, struct.pack('HHHH', 0, 0, 0, 0))))

    return _OsTerminalSize(columns=columns, lines=lines)


def _get_terminal_size_or_default():
    try:
        return int(os.environ['COLUMNS'])
    except (KeyError, ValueError):
        pass

    if sys.version_info >= (3, 3):
        query_fd = os.get_terminal_size
    else:
        query_fd = _os_get_terminal_size_pre_3_3

    for fd in (0, 1, 2):
        try:
            return query_fd(fd)
        except Exception:
            pass

    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        try:
            return query_fd(fd)
        finally:
            os.close(fd)
    except Exception:
        pass

    return _OsTerminalSize(columns=80, lines=24)


def _for_print(text):
    if sys.version_info >= (3, ):
        return text

    return text.encode(sys.stdout.encoding or 'UTF-8', 'replace')


def dump_tree(t, revision_digits, latest_revision, config,
              level=0, branch_level=-3, tag_level=-3, parent_dir=''):
    def indent_print(line_start, text):
        if config.flat_tree:
            level_text = parent_dir
        else:
            level_text = ' '*(4*level)
        if config.show_numbers:
            print('{}  {}{}'.format(line_start, level_text, _for_print(text)))
        else:
            print('{}{}'.format(level_text, _for_print(text)))

    items = ((k, v) for k, v in t.items() if k)

    if ((branch_level + 2 == level) and not config.show_branches) \
            or ((tag_level + 2 == level) and not config.show_tags) \
            or level >= config.max_depth:
        if items and config.show_dots:
            line_start = ' '*(1 + revision_digits + 2 + revision_digits + 1)
            if config.flat_tree:
                indent_print(line_start, '/[..]')
            else:
                indent_print(line_start, '[..]')
        return

    for k, (added_on_rev, last_deleted_on_rev, children) in sorted(items):
        format = '(%%%dd; %%%dd)' % (revision_digits, revision_digits)
        if last_deleted_on_rev is not None:
            last_seen_rev = last_deleted_on_rev - 1
        else:
            last_seen_rev = latest_revision
        visual_rev = format % (added_on_rev, last_seen_rev)

        indent_print(visual_rev, '/%s' % k)

        bl = branch_level
        tl = tag_level
        if k == 'branches':
            bl = level
        elif k == 'tags':
            tl = level
        dump_tree(children, revision_digits, latest_revision, config,
                  level=level + 1, branch_level=bl, tag_level=tl,
                  parent_dir='{}/{}'.format(parent_dir, k))


def dump_nick_stats(nick_stats, revision_digits, config):
    if config.show_numbers:
        format = "%%%dd (%%%dd; %%%dd)  %%s" % (revision_digits,
                                                revision_digits,
                                                revision_digits)
        for nick, (first_commit_rev, last_commit_rev, commit_count) \
                in sorted(nick_stats.items()):
            print(format % (commit_count, first_commit_rev, last_commit_rev,
                            _for_print(nick)))
    else:
        for nick, (first_commit_rev, last_commit_rev, commit_count) \
                in sorted(nick_stats.items()):
            print(_for_print(nick))



def dump_externals(externals, revision_digits, latest_revision, latest_date,
    config):
    last_path = []
    base_ident = 1 + revision_digits + 2 + revision_digits + 1 + 3 \
        if config.show_numbers else 0
    print('\nEXTERNALS\n')
    for ext_dir, ext_items in sorted(externals.items()):
        path = ext_dir.split('/')
        if config.flat_tree:
            prefix = ext_dir
        else:
            for level, name in enumerate(path):
                if level >= len(last_path) or last_path[level] != name:
                    print('{}{}'.format(' '*(base_ident + 4 * level),
                        _for_print(name)))
            prefix = ' '*(4*len(path)-2)
        last_path = path
        for (ext_url, ext_unpeg), (added_on_rev, last_deleted_on_rev, \
                remote_min_rev, remote_max_rev) in ext_items.items():
            if config.show_numbers:
                format = '(%%%dd; %%%dd)  ' \
                    % (revision_digits, revision_digits)
                if last_deleted_on_rev is not None:
                    last_seen_rev = last_deleted_on_rev - 1
                else:
                    last_seen_rev = latest_revision
                visual_rev = format % (added_on_rev, last_seen_rev)
            else:
                visual_rev = ''
            if ext_unpeg: #isinstance(remote_min_rev, datetime.datetime):
                if last_deleted_on_rev is None:
                    remote_max_rev = latest_date
                visual_remote_rev = '({}; {})'.format(
                    remote_min_rev.strftime('%Y-%m-%d'),
                    remote_max_rev.strftime('%Y-%m-%d'))
            else:
                visual_remote_rev = '({}; {})'.format(
                    remote_min_rev, remote_max_rev)
            print('{}{} {} {}  {}'.format(
                visual_rev, prefix,
                ('=>' if ext_unpeg else '->'),
                ext_url, visual_remote_rev))


def ensure_uri(text):
    import re
    svn_uri_detector = re.compile('^[A-Za-z+]+://')
    if svn_uri_detector.search(text):
        return text
    else:
        import os
        abspath = os.path.abspath(text)
        return 'file://%s' % urllib_parse_quote(abspath)


def digit_count(n):
    if n == 0:
        return 1
    assert(n > 0)
    return int(math.floor(math.log10(n)) + 1)


def hms(seconds):
    seconds = math.ceil(seconds)
    h = int(seconds / 3600)
    seconds = seconds - h*3600
    m = int(seconds / 60)
    seconds = seconds - m*60
    return h, m, seconds


def externals_parse(exts, url_root='', dir_current=''):
    exts_dict = dict() # dir -> (url, rev) where rev = None if not specified
    if not exts:
        return exts_dict
    if dir_current and not dir_current.endswith('/'):
        dir_current += '/'
    if url_root and not url_root.endswith('/'):
        url_root += '/'
    url_current = url_root + dir_current
    for l in exts.replace('\r\n', '\n').split('\n'):
        tok = shlex.split(l.strip())
        if len(tok) == 0:
            continue
        if '://' in tok[-1]:
            # pre svn-1.5 format
            ext_url = tok[-1]
            ext_dir = tok[0]
            ext_rev = None
            if len(tok) > 2:
                if tok[1].startswith('-r') and (
                        (len(tok[1]) > 2 and len(tok) == 3) or
                        (len(tok[1]) == 2 and len(tok) == 4) ):
                    ext_rev = int(tok[1][2:] if len(tok[1]>2) else tok[2])
                else:
                    print('FAILED to parse svn:externals line "{}" (1)\n'
                          .format(l), file=sys.stderr)
                    continue
        else:
            ext_rev = None
            if tok[0].startswith('-r'):
                if len(tok[0]) > 2:
                    ext_rev = int(tok[0][2:])
                    tok = tok[1:]
                elif len(tok) >= 4:
                    ext_rev = int(tok[1])
                    tok = tok[2:]
                else:
                    print('FAILED to parse svn:externals line "{}" (2)\n'
                          .format(l), file=sys.stderr)
                    continue
            if tok[0].startswith('//') or tok[0].startswith('/') \
                or tok[0].startswith('../'):
                # standard relative urls to current url
                ext_url = urllib_parse_urljoin(url_current, tok[0])
                tok = tok[1:]
            elif tok[0].startswith('^/'): # or tok[0].startswith('^
                # urls relative to root url
                ext_url = urllib_parse_urljoin(url_root, tok[0][2:])
                tok = tok[1:]
            elif '://' in tok[0]:
                ext_url = tok[0]
                tok = tok[1:]
            else:
                print('FAILED to parse svn:externals line "{}" (3)\n'
                      .format(l),file=sys.stderr)
                continue
            if len(tok) > 1:
                print('FAILED to parse svn:externals line "{}" (4)\n'
                      .format(l), file=sys.stderr)
                continue
            ext_dir = tok[0]
        if '@' in ext_url:
            if ext_rev is None:
                ext_rev = int(ext_url.split('@')[1])
            ext_url = ext_url.split('@')[0]
        exts_dict[dir_current + ext_dir] = (ext_url, ext_rev)
    #print('PARSED from {} {} {}\n{}\n=> {}\n'
    #      .format(url_root, url_current, dir_current, exts, exts_dict))
    return exts_dict


def make_progress_bar(percent, width, seconds_taken, seconds_expected):
    other_len = (6 + 1) + 2 + (1 + 8 + 1 + 9 + 1) + 3 + 1
    assert(width > 0)
    bar_content_len = width - other_len
    assert(bar_content_len >= 0)
    fill_len = int(percent * bar_content_len / 100)
    open_len = bar_content_len - fill_len
    seconds_remaining = seconds_expected - seconds_taken
    hr, mr, sr = hms(seconds_remaining)
    if hr > 99:
        hr = 99
    return ('%6.2f%%  (%02d:%02d:%02d remaining)  [%s%s]'
            % (percent, hr, mr, sr, '#'*fill_len, ' '*open_len))


def command_line():
    from svneverever.version import VERSION_STR
    parser = argparse.ArgumentParser(
            prog='svneverever',
            description='Collects path entries across SVN history',
            epilog=_EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            )
    parser.add_argument(
        '--version',
        action='version', version='%(prog)s ' + VERSION_STR)
    parser.add_argument(
        'repo_uri',
        metavar='REPOSITORY', action='store',
        help='Path or URI to SVN repository')

    modes = parser.add_argument_group('mode selection arguments')
    modes.add_argument(
        '--committers',
        dest='nick_stat_mode', action='store_true', default=False,
        help='Collect committer names instead of path information '
             '(default: disabled)')
    modes.add_argument(
        '--externals',
        dest='show_externals', action='store_true', default=False,
        help='Collect svn:externals in addition to path information '
             '(default: disabled)')

    common = parser.add_argument_group('common arguments')
    common.add_argument(
        '--no-numbers',
        dest='show_numbers', action='store_false', default=True,
        help='Hide numbers, e.g. revision ranges (default: disabled)')
    common.add_argument(
        '--no-progress',
        dest='show_progress', action='store_false', default=True,
        help='Hide progress bar (default: disabled)')
    common.add_argument(
        '--non-interactive',
        dest='interactive', action='store_false', default=True,
        help='Will not ask for input (e.g. login credentials) if required'
             ' (default: ask if required)')

    path_tree_mode = parser.add_argument_group('path tree mode arguments')
    path_tree_mode.add_argument(
        '--tags',
        dest='show_tags', action='store_true', default=False,
        help='Show content of tag folders (default: disabled)')
    path_tree_mode.add_argument(
        '--branches',
        dest='show_branches', action='store_true', default=False,
        help='Show content of branch folders (default: disabled)')
    path_tree_mode.add_argument(
        '--no-dots',
        dest='show_dots', action='store_false', default=True,
        help='Hide "[..]" omission marker (default: disabled)')
    path_tree_mode.add_argument(
        '--depth',
        dest='max_depth', metavar='DEPTH', action='store',
        type=int, default=-1,
        help='Maximum depth to print (starting at 1)')
    path_tree_mode.add_argument(
        '--flatten',
        dest='flat_tree', action='store_true', default=False,
        help='Flatten tree (default: disabled)')

    committer_mode = parser.add_argument_group('committer mode arguments')
    committer_mode.add_argument(
        '--unknown-committer',
        dest='unknown_committer_name', metavar='NAME', default='<unknown>',
        help='Committer name to use for commits'
             ' without a proper svn:author property (default: "%(default)s")')

    args = parser.parse_args()

    args.repo_uri = ensure_uri(args.repo_uri)
    if args.max_depth < 1:
        args.max_depth = six.MAXSIZE

    return args


def _login(realm, username, may_save, _tries):
    if _tries > 0:
        print('ERROR: Credentials not accepted by SVN, please try again.',
              file=sys.stderr)

    try:
        if username:
            print('Username: {}  (as requested by SVN)'.format(username),
                  file=sys.stderr)
        else:
            print('Username: ', end='', file=sys.stderr)
            username = six.moves.input('')
        password = getpass.getpass('Password: ')
        print(file=sys.stderr)
        return True, username, password, False
    except (KeyboardInterrupt, EOFError):
        print(file=sys.stderr)
        print('Operation cancelled.', file=sys.stderr)
        sys.exit(128 + signal.SIGINT)


def _create_login_callback():
    tries = 0

    def login_with_try_counter(*args, **kvargs):
        nonlocal tries

        kvargs['_tries'] = tries
        try:
            return _login(*args, **kvargs)
        finally:
            tries += 1

    return login_with_try_counter


def main():
    args = command_line()

    # Build tree from repo
    client = pysvn.Client()
    if args.interactive:
        client.callback_get_login = _create_login_callback()
    tree = dict()
    try:
        latest_revision = client.info2(
            args.repo_uri, recurse=False)[0][1]['last_changed_rev'].number
    except (pysvn.ClientError) as e:
        if str(e) == 'callback_get_login required':
            print('ERROR: SVN Repository requires login credentials'
                  '. Please run without --non-interactive switch.',
                  file=sys.stderr)
        else:
            print('ERROR: %s' % str(e), file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    print('Analyzing %d revisions...' % latest_revision, file=sys.stderr)
    width = _get_terminal_size_or_default().columns

    def indicate_progress(rev, before_work=False):
        percent = rev * 100.0 / latest_revision
        seconds_taken = time.time() - start_time
        seconds_expected = seconds_taken / float(rev) * latest_revision
        if (rev == latest_revision) and not before_work:
            percent = 100
            seconds_expected = seconds_taken
        print('\r' + make_progress_bar(percent, width,
                                       seconds_taken, seconds_expected),
              end='', file=sys.stderr)
        sys.stderr.flush()

    nick_stats = dict()
    externals_props = dict()
    externals = dict()

    for rev in xrange(1, latest_revision + 1):
        if rev == 1 and args.show_progress:
            indicate_progress(rev, before_work=True)

        last_rev_object \
            = pysvn.Revision(pysvn.opt_revision_kind.number, rev - 1)
        rev_object = pysvn.Revision(pysvn.opt_revision_kind.number, rev)
           
        if args.nick_stat_mode:
            committer_name = client.revpropget(
                'svn:author', args.repo_uri,
                rev_object)[1]
            if not committer_name:
                committer_name = args.unknown_committer_name
            (first_commit_rev, last_commit_rev, commit_count) \
                = nick_stats.get(committer_name, (None, None, 0))

            if first_commit_rev is None:
                first_commit_rev = rev
            last_commit_rev = rev
            commit_count = commit_count + 1

            nick_stats[committer_name] = (first_commit_rev, last_commit_rev,
                                          commit_count)

            if args.show_progress:
                indicate_progress(rev)
            continue

        summary = client.diff_summarize(
            args.repo_uri,
            revision1=last_rev_object,
            url_or_path2=args.repo_uri,
            revision2=rev_object,
            recurse=True,
            ignore_ancestry=True)
        #print("\n".join([str([str(e)+"="+str(v) for e,v in vars(s).items()])
        #  for s in summary]))
        def is_directory_addition(summary_entry):
            return (summary_entry.summarize_kind
                    == pysvn.diff_summarize_kind.added
                    and summary_entry.node_kind == pysvn.node_kind.dir)

        def is_directory_deletion(summary_entry):
            return (summary_entry.summarize_kind
                    == pysvn.diff_summarize_kind.delete
                    and summary_entry.node_kind == pysvn.node_kind.dir)

        def is_directory_properties(summary_entry):
            return (summary_entry.prop_changed
                    and summary_entry.node_kind == pysvn.node_kind.dir)

        dirs_added = [e.path for e in summary if is_directory_addition(e)]
        for d in dirs_added:
            sub_tree = tree
            for name in d.split('/'):
                if name not in sub_tree:
                    added_on_rev, last_deleted_on_rev, children \
                        = rev, None, dict()
                    sub_tree[name] = (added_on_rev, last_deleted_on_rev,
                                      children)
                else:
                    added_on_rev, last_deleted_on_rev, children \
                        = sub_tree[name]
                    if last_deleted_on_rev is not None:
                        sub_tree[name] = (added_on_rev, None, children)
                sub_tree = children

        def mark_deleted_recursively(sub_tree, absname, name, rev, all_dirs):
            added_on_rev, last_deleted_on_rev, children = sub_tree[name]
            all_dirs.append(absname)
            if last_deleted_on_rev is None:
                sub_tree[name] = (added_on_rev, rev, children)
                for child_name in children.keys():
                    mark_deleted_recursively(children,
                        absname + '/' + child_name, child_name, rev, all_dirs)

        dirs_deleted = [e.path for e in summary if is_directory_deletion(e)]
        all_dirs_deleted = []
        for d in dirs_deleted:
            sub_tree = tree
            comps = d.split('/')
            comps_len = len(comps)
            for i, name in enumerate(comps):
                if i == comps_len - 1:
                    mark_deleted_recursively(sub_tree, d, name, rev,
                                             all_dirs_deleted)
                else:
                    added_on_rev, last_deleted_on_rev, children \
                        = sub_tree[name]
                    sub_tree = children

        if args.show_externals:
            dirs_props = [e.path for e in summary if is_directory_properties(e)]
            commit_date = None # only get if needed (for unpeg externals)
            for d in set(dirs_props) | set(dirs_added) | set(all_dirs_deleted):
                if d in all_dirs_deleted:
                    exts = {}
                else:
                    exts = client.propget(
                        'svn:externals', args.repo_uri + '/' + d, rev_object,
                        recurse=False)
                    exts = list(exts.values())[0] if exts else ''
                    exts = externals_parse(exts, args.repo_uri, d)
                last_exts = externals_props.get(d,{})
                externals_props[d] = exts
                for ext_dir in set(last_exts) | set(exts):
                    ext_url, ext_rev = exts.get(ext_dir, (None, None))
                    last_ext_url, last_ext_rev \
                        = last_exts.get(ext_dir, (None, None))
                    ext_unpeg = bool(ext_url and not ext_rev)
                    last_ext_unpeg = bool(last_ext_url and not last_ext_rev)
                    if ext_unpeg or last_ext_unpeg:
                        if not commit_date:
                            commit_date = client.revpropget(
                                'svn:date', args.repo_uri,
                                rev_object)[1]
                            commit_date = datetime.datetime.strptime(
                                commit_date, "%Y-%m-%dT%H:%M:%S.%f%z")
                        if ext_unpeg:
                            ext_rev = commit_date
                        if last_ext_unpeg:
                            # remove one microsecond
                            last_ext_rev = commit_date \
                                - datetime.timedelta(microseconds=1)
                    if ext_dir not in externals:
                        externals[ext_dir] = dict()
                    if last_ext_url and (ext_url != last_ext_url or \
                        ext_unpeg != last_ext_unpeg):
                        # mark last dir -> URL link as deleted
                        first_commit_rev, last_deleted_on_rev, \
                            remote_min_rev, remote_max_rev \
                            = externals[ext_dir][(last_ext_url, last_ext_unpeg)]
                        last_deleted_on_rev = rev
                        externals[ext_dir][(last_ext_url, last_ext_unpeg)] \
                            = ( first_commit_rev, last_deleted_on_rev, \
                                min(remote_min_rev, last_ext_rev),
                                max(remote_max_rev, last_ext_rev) )
                    if ext_url: # add or update dir -> URL link
                        first_commit_rev, last_deleted_on_rev, \
                            remote_min_rev, remote_max_rev \
                            = externals[ext_dir].get((ext_url, ext_unpeg),
                            (rev, None, ext_rev, ext_rev))
                        last_deleted_on_rev = None
                        externals[ext_dir][(ext_url, ext_unpeg)] \
                            = ( first_commit_rev, last_deleted_on_rev, \
                                min(remote_min_rev, ext_rev),
                                max(remote_max_rev, ext_rev) )
        if args.show_progress:
            indicate_progress(rev)

    if args.show_progress:
        print(file=sys.stderr)
    print(file=sys.stderr)
    sys.stderr.flush()

    # NOTE: Leaves are files and empty directories
    if args.nick_stat_mode:
        dump_nick_stats(nick_stats, digit_count(latest_revision), config=args)
    else:
        dump_tree(tree, digit_count(latest_revision), latest_revision,
                  config=args)
        if args.show_externals:
            latest_date = client.revpropget(
                'svn:date', args.repo_uri,
                pysvn.Revision(pysvn.opt_revision_kind.number,
                latest_revision))[1]
            latest_date = datetime.datetime.strptime(
                latest_date, "%Y-%m-%dT%H:%M:%S.%f%z") if latest_date \
                else datetime.datetime.now()
            dump_externals(externals, digit_count(latest_revision),
                  latest_revision, latest_date, config=args)

if __name__ == '__main__':
    main()
