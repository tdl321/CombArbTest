#!/usr/bin/env python3
"""Super Bowl 2025 partition backtest - 32 team markets."""

import sys
sys.path.insert(0, "/root/combarbbot")
from src.backtest.tournament import TournamentMarket, PartitionTick

import logging
import glob
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/root/prediction-market-analysis/data/polymarket"


@dataclass
class TournamentMarket:
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    volume: float


@dataclass
class PartitionTick:
    timestamp: datetime
    block_number: int
    prices: dict[str, float]
    partition_sum: float
    
    @property
    def violation(self) -> float:
        return abs(self.partition_sum - 1.0)


# All 32 NFL team Super Bowl 2025 markets
SUPER_BOWL_2025_MARKETS = [
    TournamentMarket(market_id="503301", question="Will the Panthers win Super Bowl 2025?",
        yes_token_id="66714239224313823221640294891961335457341276313024833480429172445133070011668",
        no_token_id="89467374135934739834722715801896983274727814886329505374811124523387276806796", volume=139314923),
    TournamentMarket(market_id="503314", question="Will the Raiders win Super Bowl 2025?",
        yes_token_id="54892341795653479805388020158684786934714678926304197461400878299451904269209",
        no_token_id="40801206881564693043973435688806179249940112928041934634486776594277892109019", volume=123969933),
    TournamentMarket(market_id="503327", question="Will the Titans win Super Bowl 2025?",
        yes_token_id="58440649209461200280350668047868558605555236750105361041671089512720702235189",
        no_token_id="83111747692593108439667014273321109579545447301960950400715776857725380837947", volume=119500829),
    TournamentMarket(market_id="503304", question="Will the Browns win Super Bowl 2025?",
        yes_token_id="37221056259325429041732308291782104664756441972369963169173383951407900144525",
        no_token_id="63939510524223361009002618165078196129404531478278623969998606591983074882103", volume=115257552),
    TournamentMarket(market_id="503320", question="Will the Giants win Super Bowl 2025?",
        yes_token_id="1579051818984095727673421497419190670041974399782372017404834102770355812245",
        no_token_id="103025800543021453963790516518962068292483949140859491419046691780811544794182", volume=107857306),
    TournamentMarket(market_id="503329", question="Will the Patriots win Super Bowl 2025?",
        yes_token_id="31300399306949971768449961247118568299701374047026733785221770769092315493347",
        no_token_id="75600188276302107134313150350605198202630590842724839689703495381032181513643", volume=74275772),
    TournamentMarket(market_id="503298", question="Will the Falcons win Super Bowl 2025?",
        yes_token_id="114621786190778887037417697112239263106515104256005574518022428806477237896483",
        no_token_id="115378506888317167959165654380470128609283869507678462602130386826699169950824", volume=49889785),
    TournamentMarket(market_id="503297", question="Will the Cardinals win Super Bowl 2025?",
        yes_token_id="45863977636045997741544862867913468747624252555870650051585292920595598060047",
        no_token_id="115338952093275871381521406089592578146149442599494943558258177407155298888416", volume=43228740),
    TournamentMarket(market_id="503310", question="Will the Texans win Super Bowl 2025?",
        yes_token_id="40946145547892120835388934032378411687415301148304670567000395360529369472824",
        no_token_id="13767388689589800926602299149240507927219262519162147884681600566873095986070", volume=33300682),
    TournamentMarket(market_id="503323", question="Will the Steelers win Super Bowl 2025?",
        yes_token_id="21678216225487973526804977855895622426800255930641449554839164336032353007661",
        no_token_id="68625939051253896415844769011922944355276853024680068106899140956609350934097", volume=32651063),
    TournamentMarket(market_id="503328", question="Will the Commanders win Super Bowl 2025?",
        yes_token_id="16571355100327104454501190118689337347118543324830329817599252230759175472329",
        no_token_id="7958598477921247868125626874893923014274724309877711676804172989905474637451", volume=30528260),
    TournamentMarket(market_id="503311", question="Will the Colts win Super Bowl 2025?",
        yes_token_id="80309723387445340050616740130016704193903058232016305104497360785576902478879",
        no_token_id="31600906302515164942333004489127606614375591024323080987103884527557260549125", volume=26827625),
    TournamentMarket(market_id="503312", question="Will the Jaguars win Super Bowl 2025?",
        yes_token_id="4731944209859249926168586570982500445668013261402146665166541904329861129392",
        no_token_id="43757825835597398386037640359418450297784544483598407636492376826131660316069", volume=25428418),
    TournamentMarket(market_id="503317", question="Will the Dolphins win Super Bowl 2025?",
        yes_token_id="786115068931737812060412156745557005902954492066252507777390810575643513056",
        no_token_id="69074794306471999992729900663867798526262794286621096412823575974559631840025", volume=23773212),
    TournamentMarket(market_id="503325", question="Will the Seahawks win Super Bowl 2025?",
        yes_token_id="102635810088606948147311305941716999359339527059960270753019994691928431958693",
        no_token_id="71250024838480036756937380053476736902388540084527485665303774811773302565881", volume=20673323),
    TournamentMarket(market_id="503313", question="Will the Chiefs win Super Bowl 2025?",
        yes_token_id="22535833765723427929773245088435042776045949943240943368073750664192788269527",
        no_token_id="42871158580795323243941598536325925204919765750879891842520258699472618139885", volume=16403932),
    TournamentMarket(market_id="503303", question="Will the Bengals win Super Bowl 2025?",
        yes_token_id="35255375360012657117305792246090443224690221462734865996427532233323562581158",
        no_token_id="13015410011212879471677372079377323044691857849285645525795735926605419138500", volume=16154547),
    TournamentMarket(market_id="503306", question="Will the Broncos win Super Bowl 2025?",
        yes_token_id="95922065128340167030552616846757925126575450533179378041549778657025961857141",
        no_token_id="10609680115143825259228524309694215590766391139928103806365444263403409704210", volume=14117540),
    TournamentMarket(market_id="503315", question="Will the Chargers win Super Bowl 2025?",
        yes_token_id="46071143598817311049803259731300640433612842284467383372210522641051027560309",
        no_token_id="44653829712818266932854711983200076786937607665194597405694788849589088108557", volume=13879054),
    TournamentMarket(market_id="503321", question="Will the Jets win Super Bowl 2025?",
        yes_token_id="104733076087729953713940934940391391993326070110721374162122727321651009470076",
        no_token_id="101147918166067706579026756528549918571584505797810799254550351424844459653741", volume=13428656),
    TournamentMarket(market_id="503305", question="Will the Cowboys win Super Bowl 2025?",
        yes_token_id="91353054216890740335748868776987613119107738900235010213982466959971119689491",
        no_token_id="8059405312579604783993514945805451199107762053048515732379629084368237423245", volume=12217168),
    TournamentMarket(market_id="503307", question="Will the Lions win Super Bowl 2025?",
        yes_token_id="51052158557761079600060821198344572146234634299666295420235183322360518805559",
        no_token_id="24457538192307362260130901215385681960887087239978986650052790651700145579385", volume=12167588),
    TournamentMarket(market_id="503322", question="Will the Eagles win Super Bowl 2025?",
        yes_token_id="110222417228270638383974743746762302792556220380554556504458115620557107501861",
        no_token_id="34527047802979125804174050325432167077742263137042415948357405372259768531455", volume=11829712),
    TournamentMarket(market_id="503324", question="Will the 49ers win Super Bowl 2025?",
        yes_token_id="57404941070480647064900845338248984784706447590708819584066371103229440035635",
        no_token_id="105887143603794106345878722758219961934527916904279993939903253209841411968927", volume=11810114),
    TournamentMarket(market_id="503319", question="Will the Saints win Super Bowl 2025?",
        yes_token_id="3760259282641341401139581937609668002717940820088389647026890584246573710717",
        no_token_id="4148989056010685756076902340127576457087898253675700922453397399537607754636", volume=9156142),
    TournamentMarket(market_id="503316", question="Will the Rams win Super Bowl 2025?",
        yes_token_id="98530852976572810642537081503290735812837188297744380157757156848456581539250",
        no_token_id="91288597711022953726263532679116909854827468150660362561504510637160951103302", volume=9140068),
    TournamentMarket(market_id="503300", question="Will the Bills win Super Bowl 2025?",
        yes_token_id="10543796747987526217726719445503113036676541789761379932363198740436075720933",
        no_token_id="44496525088677969212608424691084899842806265405266245973131576352260105857324", volume=8862284),
    TournamentMarket(market_id="503302", question="Will the Bears win Super Bowl 2025?",
        yes_token_id="53980625471547310051004778241938112282010815940675231749571892348038485533306",
        no_token_id="115050666003448260543845677840728773264372194320429530334281913126268515492242", volume=8672879),
    TournamentMarket(market_id="503309", question="Will the Packers win Super Bowl 2025?",
        yes_token_id="7689215271552383133483004508832984274573710042218778913722716721310897959707",
        no_token_id="89002498769528768644726791543970005724643616817519213760138963002436160049197", volume=7565957),
    TournamentMarket(market_id="503326", question="Will the Buccaneers win Super Bowl 2025?",
        yes_token_id="18070149317485211603027771306394042764814941764963048490397639686705860842455",
        no_token_id="103818143323369177891595047848567825228583655301264459204859941632422400214512", volume=7542108),
    TournamentMarket(market_id="503318", question="Will the Vikings win Super Bowl 2025?",
        yes_token_id="98758685551362251798751604021121968501566060113612180716295660305468647537764",
        no_token_id="32913191084151568159961778433708656480121261742825050053826470640755980486658", volume=6450183),
    TournamentMarket(market_id="503299", question="Will the Ravens win Super Bowl 2025?",
        yes_token_id="65899342545197974464674790375677332783836387984276968232607993321656475710546",
        no_token_id="44043530156416941552422118596418820865632961462273897699074006193128135935861", volume=6399020),
]


def run_super_bowl_backtest(
    violation_threshold: float = 0.02,
    interval_minutes: int = 60,
) -> None:
    """Run backtest on Super Bowl 2025 partition."""
    data_dir = DEFAULT_DATA_DIR
    
    trades_files = [
        f for f in glob.glob(f"{data_dir}/trades/*.parquet")
        if not f.split("/")[-1].startswith("._")
    ]
    
    blocks_files = [
        f for f in glob.glob(f"{data_dir}/blocks/*.parquet")
        if not f.split("/")[-1].startswith("._")
    ]
    
    conn = duckdb.connect()
    
    # Build token -> market mapping
    token_to_market = {}
    for m in SUPER_BOWL_2025_MARKETS:
        token_to_market[m.yes_token_id] = (m.market_id, "yes", m.question[:30])
        token_to_market[m.no_token_id] = (m.market_id, "no", m.question[:30])
    
    all_tokens = []
    for m in SUPER_BOWL_2025_MARKETS:
        all_tokens.extend([m.yes_token_id, m.no_token_id])
    
    token_placeholders = ",".join(f"'{t}'" for t in all_tokens)
    
    logger.info("Querying trades for %d markets (%d tokens)...", 
                len(SUPER_BOWL_2025_MARKETS), len(all_tokens))
    
    query = f"""
        SELECT 
            t.block_number,
            t.maker_asset_id,
            t.taker_asset_id,
            t.maker_amount,
            t.taker_amount,
            b.timestamp
        FROM read_parquet({trades_files}) t
        LEFT JOIN read_parquet({blocks_files}) b ON t.block_number = b.block_number
        WHERE t.maker_asset_id IN ({token_placeholders})
           OR t.taker_asset_id IN ({token_placeholders})
        ORDER BY t.block_number
    """
    
    trades = conn.execute(query).fetchall()
    conn.close()
    
    logger.info("Found %d trades", len(trades))
    
    if not trades:
        print("No trade data found!")
        return
    
    # Initialize at equal probability (1/32 each)
    current_prices = {m.market_id: 1.0/32 for m in SUPER_BOWL_2025_MARKETS}
    ticks = []
    
    interval = timedelta(minutes=interval_minutes)
    last_snapshot_time = None
    
    for trade in trades:
        block_num, maker_asset, taker_asset, maker_amt, taker_amt, ts_str = trade
        
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
            except:
                continue
        else:
            continue
        
        # Calculate implied price from trade
        if maker_amt and taker_amt and (maker_amt + taker_amt) > 0:
            if maker_asset in token_to_market:
                mid, side, _ = token_to_market[maker_asset]
                price = taker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = max(0.001, min(0.999, price))
            elif taker_asset in token_to_market:
                mid, side, _ = token_to_market[taker_asset]
                price = maker_amt / (maker_amt + taker_amt)
                if side == "yes":
                    current_prices[mid] = max(0.001, min(0.999, price))
        
        # Create snapshot at intervals
        if last_snapshot_time is None or ts - last_snapshot_time >= interval:
            partition_sum = sum(current_prices.values())
            tick = PartitionTick(
                timestamp=ts,
                block_number=block_num,
                prices=dict(current_prices),
                partition_sum=partition_sum,
            )
            ticks.append(tick)
            last_snapshot_time = ts
    
    # Find violations
    violations = [t for t in ticks if t.violation >= violation_threshold]
    
    # Report
    print("=" * 70)
    print("SUPER BOWL 2025 PARTITION BACKTEST")
    print("=" * 70)
    print(f"\nMarkets: 32 NFL teams")
    print(f"Constraint: Exactly one team wins -> sum(P) = 1")
    
    if ticks:
        start = min(t.timestamp for t in ticks)
        end = max(t.timestamp for t in ticks)
        print(f"\nTimeline: {start} to {end}")
        print(f"Snapshots: {len(ticks)}")
        
        min_sum = min(t.partition_sum for t in ticks)
        max_sum = max(t.partition_sum for t in ticks)
        avg_sum = sum(t.partition_sum for t in ticks) / len(ticks)
        
        print(f"\nPartition Sum Analysis:")
        print(f"  Min sum: {min_sum:.4f}")
        print(f"  Max sum: {max_sum:.4f}")
        print(f"  Avg sum: {avg_sum:.4f}")
        
        print(f"\nViolations (|sum - 1| > {violation_threshold}): {len(violations)}")
        
        if violations:
            print("\nTop 10 violations:")
            top = sorted(violations, key=lambda v: -v.violation)[:10]
            for v in top:
                direction = "OVER" if v.partition_sum > 1 else "UNDER"
                print(f"  {v.timestamp}: sum={v.partition_sum:.4f} ({direction})")
                
                # Show top 5 prices for this tick
                sorted_prices = sorted(v.prices.items(), key=lambda x: -x[1])[:5]
                for mid, price in sorted_prices:
                    market = next((m for m in SUPER_BOWL_2025_MARKETS if m.market_id == mid), None)
                    team = market.question.split("Will the ")[1].split(" win")[0] if market else mid
                    print(f"    {team}: {price:.4f}")
        
        total_profit = sum(v.violation for v in violations)
        print(f"\nArbitrage Opportunities: {len(violations)}")
        print(f"Total Potential Profit: ${total_profit:.4f} per $1 wagered")
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    
    run_super_bowl_backtest(
        violation_threshold=args.threshold,
        interval_minutes=args.interval,
    )
