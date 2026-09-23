/**
 * The demo walkthrough as data. No Playwright import: this module is pure so it
 * unit-tests without a browser, and `record.mjs` is the only thing that knows
 * how to execute an action or check an assertion.
 *
 * Every figure quoted in a caption is also listed in that beat's `asserts`, so a
 * number that drifts out of the report fails the render instead of shipping.
 */

export const GEOMETRY = { width: 1440, height: 810 };
export const PAGE_RELATIVE_PATH = 'examples/vllm_subset/experiment.html';

/** Rank 17, TieringOffloadingManager._initiate_promotion, 11 proposals, 5 paper-cited. */
export const DRILL_IN_ROW = 'tbody#cand-vllm_v1_kv_offload-0009';

/**
 * The same candidate as a bare element id rather than as a selector: the value a
 * finding's proposal link carries in `data-cand`.
 *
 * Derived from DRILL_IN_ROW rather than written out, because the findings beat's whole
 * claim is that the top paper finding's first proposal landed on *the row the demo
 * opened*. A literal id there would still pass if the drill-in ever moved to another
 * candidate, and would then be proving a tie that is no longer in the demo.
 */
export const DRILL_IN_CAND = DRILL_IN_ROW.replace(/^tbody#/, '');

/**
 * The leaderboard's own section header: the h2 the reveal beat frames, on the same line
 * as the hot-modules and radial-tree toggles. There is exactly one `.sechead` in the
 * document, so it needs no `:has` to disambiguate it.
 */
export const LEADERBOARD_HEAD = '.sechead';

/**
 * The downstream-artifacts section: the three tiles the closing caption is about.
 * The section carries no id, so it is addressed by its heading. Playwright's text
 * engine matches DOM text, which `text-transform: uppercase` on `section.sec > h2`
 * does not touch, so the source spelling is the right thing to match on.
 */
export const DOWNSTREAM_SECTION = 'section.sec:has(h2:text-is("Downstream artifacts"))';

/**
 * The share of the viewport's height the radial tree must fill in the tree beat.
 *
 * Measured, not guessed: the plot renders 1186px square in the 1440x810 viewport, so
 * once it is centred it covers the full 810px (coverage 1.00), while the toggle-only
 * scroll the beat used to end on leaves just 307px of it in frame (coverage 0.38).
 * 0.90 sits well clear of both -- it fails the old framing and holds with slack for
 * whatever the layout does to the plot's size.
 */
export const TREE_VIEWPORT_COVERAGE = 0.9;

/**
 * The share of the viewport's height the opened candidate's proposal list must fill once
 * the drill-proposals travel has landed.
 *
 * Measured, not guessed: `.d-props` renders 4056px tall, so at the write-up's scroll
 * position -- where the beat starts -- only its first 242px are inside the 810px frame
 * (coverage 0.30), and with its top aligned to the top of the frame it covers all 810px
 * (coverage 1.00). 0.90 fails the starting position by a wide margin and holds with
 * slack at the destination. `inViewport` cannot ask this question of a box five times
 * the height of the viewport.
 */
export const PROPOSALS_VIEWPORT_COVERAGE = 0.9;

/**
 * The findings catalogue: the run's evidence base, and the one section that records
 * where each finding was published and whether it paid off. It carries no id of its
 * own, so it is addressed by the search box it contains.
 */
export const FINDINGS_SECTION = 'section.sec:has(#fq)';

/**
 * The catalogue's top row once it has been filtered and sorted.
 *
 * One `tbody.fnd` per finding, and the section's script sorts by re-appending *every*
 * tbody -- filtered-out ones included -- so the first tbody in document order is
 * usually a hidden one. `:visible` is what makes this the row a viewer actually sees
 * at the top of the table. The `>> nth=0` is what makes it *the* top row rather than
 * "any row": `textMatches` is satisfied by any match among the elements a selector
 * returns, so an unpinned selector would prove only that some paper finding somewhere
 * has 6 proposals, which is true however the table is ordered.
 */
export const TOP_FINDING = 'table.fnd tbody.fnd:visible >> nth=0';

/**
 * The top finding's first proposal, and the link inside it that names the candidate the
 * proposal landed on.
 *
 * `nth=0` a second time, for the same reason TOP_FINDING carries one: the claim is about
 * the *first* proposal of the *top* finding. Unpinned, `a[data-cand]` would be satisfied
 * by any proposal anywhere in the catalogue that points at the drill-in candidate, which
 * is precisely what the assertion exists to establish rather than to assume.
 */
export const TOP_FINDING_FIRST_PROP = `${TOP_FINDING} >> details.pitem >> nth=0`;
export const TOP_FINDING_CAND_LINK = `${TOP_FINDING_FIRST_PROP} >> a[data-cand]`;

/** The alignments `scrollAlign` and `scrollStepped` accept for their `block` option. */
export const ALIGN_BLOCKS = new Set(['start', 'center']);

/**
 * How far below the top of the frame a `block: 'start'` alignment has to land inside each
 * of the page's two tables, so that what it aligns is not hidden by the table's own
 * header.
 *
 * `table.lb th` and `table.fnd th` are both `position: sticky; top: 0` inside a scrolling
 * ancestor, so aligning a row to the top of its scroller parks it *underneath* that
 * header. Nothing in the DOM notices: every text and attribute assertion passes on a row
 * the header is sitting on top of, and the first recording of the findings beat lost the
 * paper's title, its `paper` tag and its publisher to exactly this, leaving a detail panel
 * belonging to a finding the frame never named.
 *
 * Measured on the rendered page at 1440x810, not estimated: the leaderboard's header box
 * is 34px tall and the catalogue's, whose `PAIR CANDIDATES` label wraps to two lines, is
 * 49px. The values below clear those with a few pixels of gap. The beats that use them
 * assert `unoccluded` on the thing the clearance exists to reveal, so a header that grows
 * fails the render instead of quietly eating a row again.
 */
export const STICKY_CLEARANCE_LB = 48;
export const STICKY_CLEARANCE_FND = 56;

/**
 * The gap above the closing section, which has no sticky header to clear and needs only
 * to not sit flush against the top edge of the frame: aligned at 0 its heading's box
 * starts at y=0 exactly.
 */
export const CLOSE_CLEARANCE = 24;

/**
 * Slack allowed when asserting where an alignment landed.
 *
 * `scrollIntoView` lands on fractional device pixels and the browser reports the box back
 * rounded, so an exact comparison against a clearance would be flaky by a pixel or two.
 * Small enough that it cannot absorb a whole heading.
 */
export const ALIGN_TOLERANCE_PX = 8;

const CUTS = new Set(['gif', 'full']);
const BOTH = ['gif', 'full'];
const FULL_ONLY = ['full'];

export const BEATS = [
  {
    id: 'the-ask',
    cuts: FULL_ONLY,
    dwell: { full: 4.0 },
    caption: { full: 'the objective: <b>reduce median TTFT and TPOT</b>' },
    actions: [
      { type: 'scrollTo', sel: '.topbar' },
      { type: 'ring', sel: '.topbar .obj' },
      { type: 'pause', ms: 600 },
    ],
    asserts: [
      { type: 'textMatches', sel: '.topbar .obj', pattern: 'median TTFT' },
      { type: 'domContains', text: 'TPOT' },
    ],
  },
  {
    id: 'cost-tiles',
    cuts: BOTH,
    dwell: { full: 4.5, gif: 2.5 },
    caption: {
      full: '<b>173</b> candidates · 60 high impact · <b>544</b> proposals · <b>$35.75</b> · 266.42M tokens, <b>85% cached</b> · 7h 07m',
      gif: '<b>173</b> candidates · <b>544</b> proposals · <b>$35.75</b> · <b>85%</b> cached',
    },
    actions: [
      { type: 'unring' },
      { type: 'scrollTo', sel: '.tiles' },
    ],
    asserts: [
      /* The candidate count the caption now leads with. The hero figure is where the
         page states it, and the label is asserted with it so the number cannot be
         witnessed by some other 173: the string occurs all over this page -- the
         "Awaiting measurement" tile is 173 too -- so a document-wide substring check
         would pass with the candidate count gone. */
      { type: 'textMatches', sel: '.hero .fig', pattern: '^173$' },
      { type: 'textMatches', sel: '.hero .figlab', pattern: 'optimisation candidates' },
      { type: 'domContains', text: '$35.75' },
      { type: 'domContains', text: '266.42M' },
      { type: 'domContains', text: '7h 07m' },
      { type: 'textMatches', sel: 'div.tile:has-text("Proposals") .n', pattern: '^544$' },
    ],
  },
  {
    /* Was the cold open, back when something outside the beat loop had already parked
       the page on the board. The overview opens the demo now and this beat follows it,
       so it scrolls itself to the leaderboard, and the id says what it reveals. */
    id: 'leaderboard-reveal',
    cuts: BOTH,
    dwell: { full: 2.5, gif: 2.0 },
    caption: {
      full: '<b>173</b> ranked optimisation candidates, from one run over <code>vllm</code>',
      gif: '<b>173</b> ranked optimisation candidates',
    },
    actions: [
      /* Not `scrollTo`. scrollIntoViewIfNeeded on #lbtable framed the rows with the
         section heading above the top of the frame, so the recorded shot was a table
         with no title -- nothing on screen said what the list was. Aligning the header
         to the top of the viewport puts "Candidate leaderboard -- 173 ranked" at the top
         of the shot with the first rows underneath it, which is the frame the caption
         is written for. */
      { type: 'scrollAlign', sel: LEADERBOARD_HEAD, block: 'start' },
    ],
    asserts: [
      { type: 'visible', sel: '#lbtable' },
      { type: 'textMatches', sel: 'h2', pattern: 'Candidate leaderboard' },
      /* The caption quotes 173, so the beat proves the board still says so -- and now
         anchored end to end, because the caption claims the *whole* heading, which is
         what the viewer reads off the top of the frame. */
      { type: 'textMatches', sel: `${LEADERBOARD_HEAD} h2`, pattern: '^Candidate leaderboard \u2014 173 ranked$' },
      /* Reading right is not the same as being on screen, and being on screen is the
         entire point of the change. The h2 rather than its container: the header box
         lands within a rounded pixel of y=0, where a subpixel of overshoot would fail
         inViewport for no reason a viewer could see, and it is the heading itself the
         caption is about. */
      { type: 'inViewport', sel: `${LEADERBOARD_HEAD} h2` },
      /* ...with the first rows beneath it, so the shot is a titled list and not a title. */
      { type: 'inViewport', sel: 'table.lb tbody.cand:visible >> nth=0' },
    ],
  },
  {
    id: 'run-shape',
    cuts: FULL_ONLY,
    dwell: { full: 5.0 },
    caption: { full: '<b>8 modules</b> walked — and every chart has a table view' },
    actions: [
      { type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-mods"]' },
      { type: 'click', sel: 'button.tvbtn[data-target="tv-mods"]' },
      { type: 'pause', ms: 800 },
    ],
    asserts: [
      { type: 'attr', sel: 'button.tvbtn[data-target="tv-mods"]', name: 'aria-expanded', equals: 'true' },
      { type: 'visible', sel: '#tv-mods' },
    ],
  },
  {
    id: 'agents',
    cuts: FULL_ONLY,
    dwell: { full: 5.0 },
    caption: { full: 'two coding agents · <b>221 of 544</b> proposals (<b>41%</b>) cite a finding' },
    actions: [{ type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-rstage"]' }],
    asserts: [
      { type: 'textMatches', sel: 'div.tile:has-text("Grounded proposals") .n', pattern: '^221$' },
      { type: 'domContains', text: 'of proposals cite a finding' },
      { type: 'domContains', text: 'Which agent did what' },
    ],
  },
  {
    id: 'source-payoff',
    cuts: FULL_ONLY,
    dwell: { full: 6.0 },
    caption: {
      full: 'what it read, and which sources <b>paid off</b>: 27 paper findings, <b>21</b> produced a proposal',
    },
    actions: [
      { type: 'scrollTo', sel: '.bar[data-tip="paper"]' },
      { type: 'hover', sel: '.bar[data-tip="paper"]' },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'visible', sel: '#tip' },
      { type: 'textMatches', sel: '#tip', pattern: '27 findings' },
      { type: 'textMatches', sel: '#tip', pattern: '21 produced a proposal' },
    ],
  },
  {
    id: 'pair-grid',
    cuts: FULL_ONLY,
    dwell: { full: 4.0 },
    caption: { full: '<b>2,064</b> pairs evaluated · <b>10.7%</b> yielded a proposal' },
    actions: [{ type: 'scrollTo', sel: 'button.tvbtn[data-target="tv-rgrid"]' }],
    asserts: [
      { type: 'domContains', text: '2,064' },
      { type: 'domContains', text: '10.7%' },
    ],
  },
  {
    id: 'board',
    cuts: BOTH,
    dwell: { full: 8.0, gif: 3.0 },
    caption: {
      full: 'hot modules: <b>4 of 8</b> modules hold <b>41 of the 60</b> high-impact candidates',
      gif: 'hot modules: <b>4 of 8</b> hold <b>41 of 60</b> high-impact',
    },
    actions: [
      { type: 'scrollTo', sel: '#hotbtn' },
      { type: 'ring', sel: '#hotbtn' },
      { type: 'click', sel: '#hotbtn' },
      { type: 'pause', ms: 1200 },
      { type: 'unring' },
      { type: 'select', sel: '#mod', value: 'vllm/v1/kv_offload' },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'attr', sel: '#hotbtn', name: 'aria-pressed', equals: 'true' },
      /* #lbnote is a transient toast, set only by the deep-link reveal and cleared
         by any filter change -- including this beat's own select -- so it can never
         witness hot mode. #count is the live proof the filter took effect, and the
         caption's own figures are asserted against the prose that states them. */
      { type: 'textMatches', sel: '#count', pattern: '^\\d+ / 173 shown$' },
      { type: 'domContains', text: '4 of 8 modules, holding 41 of the 60 high-impact candidates' },
      { type: 'visible', sel: DRILL_IN_ROW },
    ],
  },
  {
    id: 'drill-writeup',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 2.5 },
    caption: {
      full: 'every candidate is a full write-up — what it is, how it works today, why it is a candidate',
      gif: 'every candidate is a full write-up',
    },
    actions: [
      { type: 'scrollTo', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'click', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'pause', ms: 900 },
    ],
    asserts: [
      { type: 'visible', sel: `${DRILL_IN_ROW} tr.detail` },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-grid`, pattern: 'How it works today' },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .sym`, pattern: '_initiate_promotion' },
    ],
  },
  {
    id: 'drill-proposals',
    cuts: BOTH,
    /* gif 3.0 -> 3.5. The travel is now driven in ten 90ms hops plus a settle rather than
       jumped, which costs ~650ms more than the scrollIntoViewIfNeeded it replaced; at 3.0
       the gif cut overran the beat and the arrival was cut into the caption's fade-out.
       The full dwell already had the headroom, so it does not move. */
    dwell: { full: 4.0, gif: 3.5 },
    caption: {
      full: '<b>11 proposals</b>, 5 grounded in papers — arXiv, USENIX, AAAI',
      gif: '<b>11 proposals</b>, 5 from papers',
    },
    actions: [
      /* Stepped, not scrollIntoViewIfNeeded. The write-up and the proposals are two ends
         of one expanded row, and an instant cut between them reads as a new screen rather
         than as more of the same candidate; seeing the page travel is what tells the
         viewer the proposals belong to the candidate they just read about. The distance
         is ~568px -- 451px of it inside the leaderboard's own max-height scroller and
         117px the document -- driven in ten hops, so the recording holds frames at
         intermediate scroll positions instead of one before and one after.

         The clearance matters here as much as the travel: aligned flush, the list's own
         "11 proposals" heading -- the figure the caption leads with -- landed at y -1..15,
         behind the leaderboard's sticky header, so the beat arrived at a wall of proposal
         prose with the count that explains it covered up. */
      {
        type: 'scrollStepped',
        sel: `${DRILL_IN_ROW} .d-props`,
        block: 'start',
        marginTop: STICKY_CLEARANCE_LB,
        steps: 10,
        stepMs: 90,
      },
      { type: 'pause', ms: 600 },
    ],
    asserts: [
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-props`, pattern: '11 proposals' },
      { type: 'textMatches', sel: `${DRILL_IN_ROW} .d-props`, pattern: 'from paper:' },
      { type: 'minChildren', sel: `${DRILL_IN_ROW} .d-props`, n: 11 },
      /* The three above read the DOM at any scroll position at all, so none of them can
         witness where the travel ended. This one can: the list is 4056px tall, which
         `inViewport` can never accept, so the question worth asking is how much of the
         frame it fills. 0.30 at the write-up, 1.00 once it has arrived. */
      { type: 'viewportCoverage', sel: `${DRILL_IN_ROW} .d-props`, minFraction: PROPOSALS_VIEWPORT_COVERAGE },
      /* The caption's "11 proposals" is legible where the travel arrived, not behind the
         leaderboard's sticky header. Coverage cannot ask this: the list fills the frame
         either way, which is precisely how the heading went missing unnoticed.
         `inViewport` is not the guard here either, and neither is any text assertion: flush
         against the scroller the h4's box was y -1..15, which the text assertions above
         pass regardless and which inViewport rejects only by the one pixel that happened
         to hang off the top edge. `unoccluded` asks the browser what is painted at the
         heading's centre, so the leaderboard's sticky `th` covering it fails the render
         however nearly it fits. */
      { type: 'unoccluded', sel: `${DRILL_IN_ROW} .d-props > h4` },
    ],
  },
  {
    id: 'radial-tree',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 1.5 },
    caption: {
      full: 'or see all <b>173</b> as a tree',
      gif: 'or see all <b>173</b> as a tree',
    },
    actions: [
      { type: 'click', sel: `${DRILL_IN_ROW} tr.row` },
      { type: 'select', sel: '#mod', value: '' },
      { type: 'scrollTo', sel: '#lbview' },
      /* Hot mode is still on from the board beat, and the tree obeys the filters: it
         drew 86 candidates under a hub reading "hot modules" while the caption said all
         173. Invisible while the tree hung off the bottom of the frame; the moment the
         frame is filled the hub is the middle of the shot, so the filter is cleared and
         the tree really is the whole run. */
      { type: 'click', sel: '#hotbtn' },
      { type: 'click', sel: '#lbview' },
      /* The toggle sits in the section header, so the beat's scrollTo leaves the tree
         hanging off the bottom of the frame. It is drawn ~1186px tall in a 810px
         viewport, so there is no scroll position that contains it: centring it is what
         fills the frame, and it puts the hub -- the "173 candidates" label -- in the
         middle of the shot. The tree is drawn at its final size by the time the toggle's
         click resolves, so this comes *before* the beat's pause: framed last, the
         centred tree would be on screen only for the fade-out at the end of a beat whose
         choreography has already spent the dwell. */
      { type: 'scrollAlign', sel: '#rtplot svg', block: 'center' },
      { type: 'pause', ms: 1400 },
    ],
    asserts: [
      { type: 'visible', sel: '#lbtree' },
      { type: 'minChildren', sel: '#rtplot svg', n: 1 },
      /* Not inViewport: that demands the whole box be inside the frame, which a tree
         taller than the viewport can never satisfy. This asks the question the beat is
         actually about -- how much of the frame the tree fills. */
      { type: 'viewportCoverage', sel: '#rtplot svg', minFraction: TREE_VIEWPORT_COVERAGE },
      /* The caption says "all 173", so the beat proves no filter is still narrowing the
         tree. #count is the live count and keeps its text while the table is hidden. */
      { type: 'textMatches', sel: '#count', pattern: '^173 / 173 shown$' },
    ],
  },
  {
    /**
     * The evidence base -- and the loop it closes.
     *
     * It sits here, after the tree, because the demo has by now shown the whole run and
     * one candidate out of it in detail; this is where that candidate's provenance is.
     * The catalogue is not interesting as a list. It is interesting because of the two
     * columns it exists for -- where a finding was published, and what it produced -- so
     * the choreography asks the payoff question outright: narrow to the 27 findings that
     * came from papers, rank them by how many proposals they produced, and open the top
     * one.
     *
     * Opening it is the substance of the beat, not a flourish. The expanded detail names
     * the technique extracted from the paper, quotes the evidence it was extracted from,
     * and then lists the proposals that technique produced -- and the first of those
     * proposals is rank #17, TieringOffloadingManager._initiate_promotion: the candidate
     * the two drill-in beats just walked through. So the shot holds the whole loop the
     * run is for -- paper, technique, proposal, ranked candidate -- in one frame, instead
     * of ending on a ranked table and leaving the viewer to take the connection on
     * trust. The asserts below tie that link to DRILL_IN_ROW in code rather than to a
     * copy of its id, so the beat cannot keep passing if the drill-in ever moves.
     *
     * The first `details.pitem` is deliberately left shut. Its "why the technique applied
     * here" body is ~250 words of 12.5px prose -- unreadable at the size the gif is
     * watched at; opening it would push the candidate link, which is the thing that
     * closes the loop, a screenful below the "6 proposals it produced" heading the
     * caption quotes; and clicking the summary means clicking within a few pixels of the
     * `a[data-cand]` itself, whose delegated handler jumps the page back to the
     * leaderboard. The summary line already renders everything the caption claims: the
     * rank, the impact, and the symbol.
     *
     * The `#fq` search demonstration this beat used to do is gone: `actions` is shared by
     * both cuts, and a fill-pause-clear costs ~1.9s of cursor travel and hold.
     *
     * No pause between the two selects. Moving the synthetic cursor from #fsrc to
     * #fsort is itself ~0.9s, so the filtered-but-unsorted table is already held long
     * enough to read before the sort lands on top of it.
     */
    id: 'findings',
    cuts: BOTH,
    /* 4.0 -> 6.0, in both cuts. The beat gained a click and a re-frame after the two
       selects, so its choreography now spends ~5.3s of the dwell-as-deadline budget; at
       4.0 every beat would have overrun and the opened finding would have been on screen
       only for the caption's fade-out. 6.0 holds the expanded row, with the candidate
       link in it, for over a second after the travel settles -- long enough to read a
       rank and a symbol name. */
    dwell: { full: 6.0, gif: 6.0 },
    caption: {
      full: 'the evidence base: <b>97</b> findings, tracked to where each was published (<b>46</b> sites) and whether it paid off — rank the <b>27</b> from papers by payoff and open the top one: this paper taught one technique, which produced <b>6</b> proposals, the first of them rank <b>#17</b> — the candidate we just drilled into',
      gif: '<b>97</b> findings, <b>46</b> sites — the <b>27</b> from papers, ranked by payoff. the top paper produced <b>6</b> proposals; the first is rank <b>#17</b>, the candidate we opened',
    },
    actions: [
      { type: 'scrollTo', sel: '#fq' },
      { type: 'select', sel: '#fsrc', value: 'paper' },
      { type: 'select', sel: '#fsort', value: 'props:-1' },
      /* The title cell, not the row. The row's own centre lands on the outbound source
         link -- target=_blank, and the section's handler deliberately does not treat it
         as the expander -- so clicking the row's box would open paperity.org in a new
         tab instead of expanding the finding. */
      { type: 'click', sel: `${TOP_FINDING} >> tr.frow .ftitle` },
      /* The detail is ~700px of content unfolding below a row that sits two thirds of the
         way down the frame, so without this the expansion happens off screen. 'start' on
         the whole tbody rather than on the detail's own body: aligning `.fbody` puts the
         "technique extracted" and "supporting evidence" headings underneath the table's
         sticky `th`, and those headings are half of what the shot is for.

         With the clearance, and not without it. Aligned flush, the tbody's own first row
         went under that 49px header -- title at y 10..29, `paper` tag at 12..33 -- so the
         recorded frame held a detail panel whose finding it never named, and the beat
         claiming to open a *paper* showed no paper. At 56 the title, the tag and the
         publisher sit clear of the header and the candidate link still lands at y 398,
         well above the caption. */
      { type: 'scrollAlign', sel: TOP_FINDING, block: 'start', marginTop: STICKY_CLEARANCE_FND },
      { type: 'pause', ms: 600 },
    ],
    asserts: [
      /* Two of the caption's figures at once, off the catalogue's own counter: 27 of
         97 shown. This replaces a document-wide `domContains '97 findings'`, which the
         page satisfies from several places and which would therefore have passed with
         the whole catalogue deleted. Anchored, so a wider filter cannot satisfy it. */
      { type: 'textMatches', sel: '#fcount', pattern: '^27 / 97 shown$' },
      /* The section's own 97, from the section's own heading. */
      { type: 'textMatches', sel: `${FINDINGS_SECTION} > h2`, pattern: '97 findings' },
      /* 46 sites is the figure the sort control states it under, so that is where the
         caption's 46 is witnessed rather than anywhere in the document. */
      { type: 'textMatches', sel: '#fsort option[value="host:1"]', pattern: '46 sites' },
      /* The top row after the sort really is a paper finding with 6 proposals. The two
         data attributes are the sort key and the filter key themselves; the two
         rendered cells are what a viewer reads off the frame, so both are checked. The
         highest proposal count in the whole catalogue is 7, on a finding that is not a
         paper, so a sort that quietly ignored the filter would fail these. */
      { type: 'attr', sel: TOP_FINDING, name: 'data-src', equals: 'paper' },
      { type: 'attr', sel: TOP_FINDING, name: 'data-props', equals: '6' },
      { type: 'textMatches', sel: `${TOP_FINDING} >> tr.frow .tag`, pattern: '^paper$' },
      { type: 'textMatches', sel: `${TOP_FINDING} >> tr.frow td.r >> nth=1`, pattern: '^6$' },

      /* --- the click landed, and the frame shows what the caption says it shows --- */

      /* `tr.fdetail` exists in the DOM whether or not the row is open, so this is
         `visible` and not `minChildren`: it is the assertion that fails if the click
         missed the expander or hit the source link instead. */
      { type: 'visible', sel: `${TOP_FINDING} >> tr.fdetail` },
      /* The two things the detail says about the paper: what was taken from it, and the
         quoted passage it was taken from. Both headings, scoped to this row's detail --
         the catalogue has 97 of each. */
      { type: 'textMatches', sel: `${TOP_FINDING} >> tr.fdetail h4`, pattern: '^Technique extracted$' },
      { type: 'textMatches', sel: `${TOP_FINDING} >> tr.fdetail h4`, pattern: '^Supporting evidence$' },
      /* The caption's "produced 6 proposals" as the frame renders it, anchored: the
         viewer reads this heading, so the caption and the heading must agree word for
         word rather than merely both mention a 6. */
      { type: 'textMatches', sel: `${TOP_FINDING} >> .fprops > h4`, pattern: '^6 proposals it produced$' },
      /* ...and the first of those proposals is the candidate the demo just drilled into.
         `equals: DRILL_IN_CAND` is derived from DRILL_IN_ROW rather than written out, so
         retargeting the drill-in retargets this too -- a literal id here would keep
         passing while the beats pointed at different candidates, which is exactly the
         drift that would make the caption a lie. */
      { type: 'attr', sel: TOP_FINDING_CAND_LINK, name: 'data-cand', equals: DRILL_IN_CAND },
      /* The rank and the symbol as they are rendered on the summary line, because
         `data-cand` is an attribute and a viewer cannot read attributes. These are what
         make the tie legible in the frame rather than merely true in the DOM. */
      { type: 'textMatches', sel: `${TOP_FINDING_FIRST_PROP} >> .prank`, pattern: '^#17$' },
      { type: 'textMatches', sel: TOP_FINDING_CAND_LINK, pattern: '_initiate_promotion' },
      /* And it is in frame. The whole point of the re-frame is that the link is legible
         in the recording, which no amount of DOM truth can witness. */
      { type: 'inViewport', sel: TOP_FINDING_CAND_LINK },
      /* And so is the finding it belongs to. These three cells name it: the paper's
         title, the `paper` tag that says what kind of source it is, and the site it was
         published on. Every assertion above is satisfied by a row the catalogue's sticky
         header is sitting on top of -- which is what the first recording of this beat was,
         an anonymous detail panel -- so what has to be asserted is that it is not. */
      { type: 'unoccluded', sel: `${TOP_FINDING} >> tr.frow .ftitle` },
      { type: 'unoccluded', sel: `${TOP_FINDING} >> tr.frow .tag` },
      { type: 'unoccluded', sel: `${TOP_FINDING} >> tr.frow .fhost` },
    ],
  },
  {
    id: 'close',
    cuts: BOTH,
    dwell: { full: 5.0, gif: 2.0 },
    caption: {
      full: '3 evolve scaffolds · 2 patches applied · <b>nothing measured yet</b> — the page says so',
      gif: '3 evolve scaffolds · 2 applied · <b>nothing measured yet</b>',
    },
    actions: [
      /* Not `scrollTo`, which was a no-op here from the moment the findings beat moved
         in front of this one. The catalogue sits directly above the downstream section, so
         the section was always already part-way into frame, and scrollIntoViewIfNeeded
         does nothing at all in that case: the closing shot stayed parked on the previous
         beat's expanded finding with the three tiles crowded into the last 130px of the
         frame, under a caption that lay across them. Measured there, the section's box was
         y 681..810: flush against the bottom edge, and inside the frame by a fraction of a
         pixel -- which is all `inViewport` asks, and is why it passed in the recording. */
      { type: 'scrollAlign', sel: DOWNSTREAM_SECTION, block: 'start', marginTop: CLOSE_CLEARANCE },
    ],
    asserts: [
      /* The two domContains guard the caption's figures wherever they sit in the
         document. Neither can witness that the closing frame is actually looking at the
         tiles, which is what the caption claims, so the two geometry assertions do: the
         section is in frame, and the frame is anchored on it. */
      { type: 'inViewport', sel: DOWNSTREAM_SECTION },
      /* The section fitting is not the same as the section being the shot, and the
         difference is the whole defect: a 129px section fits from anywhere in the frame,
         including the bottom strip it was stranded in, under the caption, below most of
         the previous beat's expanded finding. inViewport cannot tell those apart, so this
         asserts where the alignment actually landed. */
      { type: 'nearTop', sel: DOWNSTREAM_SECTION, maxY: CLOSE_CLEARANCE + ALIGN_TOLERANCE_PX },
      { type: 'domContains', text: 'Awaiting measurement' },
      { type: 'domContains', text: 'no candidate has a measured speedup yet' },
    ],
  },
];

export function beatsForCut(cut) {
  if (!CUTS.has(cut)) throw new Error(`unknown cut: ${cut}`);
  return BEATS.filter((b) => b.cuts.includes(cut));
}

export function dwellFor(beat, cut) {
  return beat.dwell[cut] ?? beat.dwell.full;
}

export function captionFor(beat, cut) {
  return beat.caption[cut] ?? beat.caption.full ?? null;
}

export function totalDuration(cut) {
  const total = beatsForCut(cut).reduce((sum, b) => sum + dwellFor(b, cut), 0);
  return Math.round(total * 10) / 10;
}

/**
 * Everything checkable about one beat on its own.
 *
 * Separate from `validateStoryboard` so a rule can be tested by handing it a beat that
 * breaks it. The alternative is a test that re-states the rule over the real BEATS, which
 * proves the storyboard is currently clean and proves nothing about the check: it passes
 * just as well when the check has been deleted.
 */
export function beatProblems(beat) {
  const problems = [];
  if (!beat.caption.full) problems.push(`${beat.id}: missing full caption`);
  if (typeof beat.dwell.full !== 'number') problems.push(`${beat.id}: missing full dwell`);
  if (beat.asserts.length === 0) problems.push(`${beat.id}: no assertions`);

  for (const a of beat.asserts) {
    if (a.type === 'textMatches') {
      try {
        new RegExp(a.pattern);
      } catch {
        problems.push(`${beat.id}: uncompilable pattern ${a.pattern}`);
      }
    }
    /* Without a bound, `box.y > undefined` is false and the assertion passes from
       anywhere in the frame -- the vacuous assertion it was added to replace. */
    if (a.type === 'nearTop' && !(Number.isFinite(a.maxY) && a.maxY >= 0)) {
      problems.push(`${beat.id}: nearTop needs a non-negative maxY, got ${a.maxY}`);
    }
  }
  for (const a of beat.actions) {
    if (a.sel === '#theme') problems.push(`${beat.id}: clicks the theme toggle`);
    /* The two scroll actions that take an alignment take the same alignments, and a
       typo in one would otherwise reach record.mjs as a silent default. */
    if (a.type === 'scrollAlign' || a.type === 'scrollStepped') {
      if (!ALIGN_BLOCKS.has(a.block)) problems.push(`${beat.id}: ${a.type} block ${a.block}`);
      /* `marginTop` is optional and defaults to 0, so a misspelled or negative one
         would reach the recorder as "align flush" -- the framing this exists to stop,
         and the one failure mode nothing else here would notice. */
      if (a.marginTop !== undefined && !(Number.isFinite(a.marginTop) && a.marginTop >= 0)) {
        problems.push(`${beat.id}: ${a.type} marginTop must be a non-negative number, got ${a.marginTop}`);
      }
    }
    if (a.type === 'scrollStepped') {
      if (!(Number.isInteger(a.steps) && a.steps >= 2)) {
        problems.push(`${beat.id}: scrollStepped needs at least 2 steps, got ${a.steps}`);
      }
      if (!(typeof a.stepMs === 'number' && a.stepMs > 0)) {
        problems.push(`${beat.id}: scrollStepped needs a positive stepMs, got ${a.stepMs}`);
      }
    }
  }
  if (beat.cuts.includes('gif')) {
    if (!beat.caption.gif) problems.push(`${beat.id}: in gif cut but has no gif caption`);
    if (typeof beat.dwell.gif !== 'number') problems.push(`${beat.id}: in gif cut but has no gif dwell`);
  }
  return problems;
}

/** Returns a list of problems. Empty means the storyboard is internally consistent. */
export function validateStoryboard() {
  const problems = [];
  const seen = new Set();

  for (const beat of BEATS) {
    if (seen.has(beat.id)) problems.push(`duplicate beat id: ${beat.id}`);
    seen.add(beat.id);
    problems.push(...beatProblems(beat));
  }

  /* 64.0, not 62.0: the findings beat's dwell went 4.0 -> 6.0 to pay for opening the top
     paper finding and re-framing it. 23.0, not 20.5: the same +2.0, plus the +0.5 the
     stepped scroll in drill-proposals costs the short cut. */
  if (totalDuration('full') !== 64.0) problems.push(`full cut is ${totalDuration('full')}s, want 64.0s`);
  if (totalDuration('gif') !== 23.0) problems.push(`gif cut is ${totalDuration('gif')}s, want 23.0s`);

  return problems;
}
