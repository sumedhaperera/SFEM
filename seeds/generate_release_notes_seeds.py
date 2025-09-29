#!/usr/bin/env python3
import sys
releases = [int(x) for x in (sys.argv[1:] or ["258","256","252","250","248","244"])]
topics = [
  "release-notes.rn_automate_flow_builder.htm",
  "release-notes.rn_automate_flow_builder_screen_flows.htm",
  "release-notes.rn_automate_flow_release_update.htm",
]
tpl = "https://help.salesforce.com/s/articleView?id={t}&language=en_US&release={r}&type=5"
for r in releases:
  for t in topics:
    print(tpl.format(t=t, r=r))

