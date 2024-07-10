import os


def get_loc_file(bug_id, perfect):
    dirname = os.path.dirname(__file__)
    if perfect:
        loc_file = '../Datasets/D4J/location/%s.buggy.lines' % (bug_id)
    else:
        loc_file = '../location/ochiai/%s/%s.txt' % (bug_id.split("-")[0].lower(), bug_id.split("-")[1])
    loc_file = os.path.join(dirname, loc_file)
    if os.path.isfile(loc_file):
        return loc_file
    else:
        print(loc_file)
        return ""


# grab location info from bug_id given
# perfect fault localization returns 1 line, top n gets top n lines for non-perfect FL (40 = decoder top n)
def get_location(bug_id, perfect=True, top_n=40):
    location = []
    loc_file = get_loc_file(bug_id, perfect)
    if loc_file == "":
        return location
    lines = open(loc_file, 'r').readlines()
    for loc_line in lines:
        loc_line = loc_line.split("||")[0]  # take first line in lump
        classname, line_id, line = loc_line.split('#')
        location.append(int(line_id))
    return location[:top_n]
