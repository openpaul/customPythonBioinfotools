from cstbioinfo.airr import perfect_paired, cast_to_pairs
import polars as pl


def get_clone_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell_id": [
                "cell1",
                "cell1",
                "cell2",
                "cell2",
                "cell3",
                "cell3",
                "cell3",
            ],
            "locus": [
                "IGH",
                "IGK",
                "IGH",
                "IGL",
                "IGH",
                "IGK",
                "IGL",
            ],
            "umi_count": [10, 5, 8, 12, 15, 7, 12],
            "sequences": ["H", "L", "H", "L", "H", "X", "L"],
        }
    )


def test_perfect_paired():
    df = get_clone_table()
    paired = perfect_paired(df, cell_column="cell_id", umi_count_column="umi_count")
    print(paired)
    assert paired.height == 3 * 2  # 3 cells with perfect pairs
    # ensure X is not in the result
    assert "X" not in paired.get_column("sequences").to_list()


def test_cast_to_pairs():
    df = get_clone_table()
    paired = cast_to_pairs(
        df, cell_column="cell_id", umi_count_column="umi_count", only_pairs=True
    ).sort(["umi_count_heavy", "umi_count_light"], descending=True)

    assert paired.height == 3  # 3 cells with perfect pairs

    umi_count_heavy_expected = [15, 10, 8]
    umi_count_light_expected = [12, 5, 12]
    assert paired.get_column("umi_count_heavy").to_list() == umi_count_heavy_expected
    assert paired.get_column("umi_count_light").to_list() == umi_count_light_expected


def test_cast_to_all_combinations():
    df = get_clone_table()
    all_comb = cast_to_pairs(
        df, cell_column="cell_id", umi_count_column="umi_count", only_pairs=False
    ).sort(["umi_count_heavy", "umi_count_light"], descending=True)

    assert (
        all_comb.height == 4
    )  # 3 cells with perfect pairs + 1 extra combination in cell3

    umi_count_heavy_expected = [15, 15, 10, 8]
    umi_count_light_expected = [12, 7, 5, 12]
    assert all_comb.get_column("umi_count_heavy").to_list() == umi_count_heavy_expected
    assert all_comb.get_column("umi_count_light").to_list() == umi_count_light_expected


def get_complicated_clone_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cell_id": [
                "cell1",
                "cell1",
                "cell2",
                "cell2",
                "cell3",
                "cell3",
                "cell3",
                "cell4",
                "cell5",
                "cell6",
                "cell6",
                "cell6",
                "cell6",
            ],
            "locus": [
                "IGH",
                "IGK",
                "IGH",
                "IGL",
                "IGH",
                "IGK",
                "IGL",
                "IGK",
                "IGH",
                "IGH",
                "IGH",
                "IGK",
                "IGL",
            ],
            "umi_count": [10, 5, 8, 12, 15, 7, 12, 1, 2, 5, 11, 3, 4],
            "sequences": [
                "H",
                "L",
                "H",
                "L",
                "H",
                "X",
                "L",
                "X",
                "X",
                "X",
                "H",
                "X",
                "L",
            ],
        }
    )


def test_perfect_paired_complicated():
    df = get_complicated_clone_table()
    paired = perfect_paired(df, cell_column="cell_id", umi_count_column="umi_count")
    print(paired)
    assert paired.height == 4 * 2  # 4 cells with perfect pairs
    # ensure X is not in the result
    assert "X" not in paired.get_column("sequences").to_list()


def test_cast_to_pairs_complicated():
    df = get_complicated_clone_table()
    paired = cast_to_pairs(
        df, cell_column="cell_id", umi_count_column="umi_count", only_pairs=True
    ).sort(["umi_count_heavy", "umi_count_light", "cell_id"], descending=True)

    assert paired.height == 4  # 4 cells with perfect pairs
    cell_id_expected = ["cell3", "cell6", "cell1", "cell2"]
    umi_count_heavy_expected = [15, 11, 10, 8]
    umi_count_light_expected = [12, 4, 5, 12]
    assert paired.get_column("cell_id").to_list() == cell_id_expected
    assert paired.get_column("umi_count_heavy").to_list() == umi_count_heavy_expected
    assert paired.get_column("umi_count_light").to_list() == umi_count_light_expected


def test_cast_to_all_combinations_complicated():
    df = get_complicated_clone_table()
    all_comb = cast_to_pairs(
        df, cell_column="cell_id", umi_count_column="umi_count", only_pairs=False
    ).sort(["umi_count_heavy", "umi_count_light"], descending=True)
    print(all_comb)
    assert (
        all_comb.height == 8
    )  # 4 cells with perfect pairs + 1 extra combination in cell3 + 3 extra in cell6

    umi_count_heavy_expected = [15, 15, 11, 11, 10, 8, 5, 5]
    umi_count_light_expected = [12, 7, 4, 3, 5, 12, 4, 3]
    assert all_comb.get_column("umi_count_heavy").to_list() == umi_count_heavy_expected
    assert all_comb.get_column("umi_count_light").to_list() == umi_count_light_expected
