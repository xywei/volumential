import volumential.list1_gallery as l1g


def test_1d():
    l1g1d, _, _ = l1g.generate_list1_gallery(1)
    assert (len(l1g1d) == 6 + 1)

    distinct_numbers = set()
    for vec in l1g1d:
        for l in vec:
            distinct_numbers.add(l)

    assert (len(distinct_numbers) == 7)
    assert (max(distinct_numbers) == 6)
    assert (min(distinct_numbers) == -6)


def test_2d():
    l1g2d, _, _ = l1g.generate_list1_gallery(2)
    assert (len(l1g2d) == 8 + 12 + 12 + 1)

    distinct_numbers = set()
    for vec in l1g2d:
        for l in vec:
            distinct_numbers.add(l)

    assert (len(distinct_numbers) == 11)
    assert (max(distinct_numbers) == 6)
    assert (min(distinct_numbers) == -6)


def test_3d():
    l1g3d, _, _ = l1g.generate_list1_gallery(3)

    distinct_numbers = set()
    for vec in l1g3d:
        for l in vec:
            distinct_numbers.add(l)

    assert (max(distinct_numbers) == 6)
    assert (min(distinct_numbers) == -6)
