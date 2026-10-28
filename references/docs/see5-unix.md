# C5.0: An Informal Tutorial

Welcome to C5.0, a system that extracts informative patterns from data.
The following sections show how to prepare data files for C5.0
and illustrate the options for using the system.

In this tutorial, file names and C5.0 input appear in
`blue fixed-width font`
while file extensions and other general forms
are shown highlighted in green.

* **[Preparing Data for C5.0](#DATA)**
  + [Application files](#files)+ [Names file](#.names)
      - [What's in a name?](#name)- [Specifying the classes](#classes)- [Explicitly-defined attributes](#explicit)- [Attributes defined by formulas](#implicit)- [Dates, times, and timestamps](#dates)- [Selecting the attributes that can appear in classifiers](#select)+ [Data file](#.data)+ [Test and cases files (optional)](#.test)+ [Costs file (optional)](#.costs)* **[Constructing Classifiers](#CLASSIFIERS)**
    + [Decision trees](#TREES)+ [Evaluation](#EVALUATION)+ [Discrete value subsets](#SUBSETS)+ [Rulesets](#RULES)
            - [Rule utility ordering](#utility)+ [Boosting](#BOOSTING)+ [Winnowing attributes](#WINNOWING)+ [Soft thresholds](#SOFT)+ [Advanced pruning options](#OTHER)+ [Sampling from large datasets](#SAMPLE)+ [Cross-validation trials](#XVAL)+ [Differential misclassification costs](#COSTS)+ [Weighting individual cases](#CASEWEIGHT)* **[Using Classifiers](#USE)*** **[Segmentation fault errors](#SEGFAULT)*** **[Linux GUI](#LINUXGUI)*** **[Linking to Other Programs](#HOOKS)*** **[Appendix: Summary of Options](#SUMMARY)**

---

## Preparing Data for C5.0

We will illustrate C5.0 using a medical application -- mining a database
of thyroid assays from the Garvan Institute of Medical Research,
Sydney, to construct diagnostic rules for hypothyroidism.
Each case concerns a single referral and contains information on the
source of the referral, assays requested, patient data, and
referring physician's comments.
Here are three examples:

```
        <u>Attribute</u>                 <u>Case 1</u>    <u>Case 2</u>    <u>Case 3</u>    .....

	age                       41        23        46
	sex                       F         F         M
	on thyroxine              f         f         f
	query on thyroxine        f         f         f
	on antithyroid medication f         f         f
	sick                      f         f         f
	pregnant                  f         f         not applicable
	thyroid surgery           f         f         f
	I131 treatment            f         f         f
	query hypothyroid         f         f         f
	query hyperthyroid        f         f         f
	lithium                   f         f         f
	tumor                     f         f         f
	goitre                    f         f         f
	hypopituitary             f         f         f
	psych                     f         f         f
	TSH                       1.3       4.1       0.98
	T3                        2.5       2         unknown
	TT4                       125       102       109
	T4U                       1.14      unknown   0.91
	FTI                       109       unknown   unknown
	referral source           SVHC      other     other
	diagnosis                 negative  negative  negative
	ID                        3733      1442      2965
```

This is exactly the sort of task for which C5.0 was designed.
Each case belongs to one of a small number of mutually exclusive classes
(negative, primary, secondary, compensated).
Properties of every case that *may* be relevant to its
class are provided, although some cases may
have unknown or non-applicable values for some attributes.
There are 24 attributes in this example, but C5.0 can deal with
any number of attributes.

C5.0's job is to find how to predict a case's class from
the values of the other attributes.
C5.0 does this by constructing a *classifier* that makes
this prediction. As we will see, C5.0 can construct classifiers expressed
as *decision trees* or as sets of *rules*.

### Application files

Every C5.0 application has a short name called
a filestem;
we will use the filestem `hypothyroid`
for this illustration.
All files read or written by C5.0 for an application
have names of the form
filestem`.`extension,
where filestem identifies the application and
extension describes the contents of the file.

Here is a summary table of the extensions used by C5.0 (to
be described in later sections):

|  |  |  |
| --- | --- | --- |
| names description of the application's attributes [required] | | |
| data cases used to generate a classifier [required] | | |
| test unseen cases used to test a classifier [optional] | | |
| cases cases to be classified subsequently [optional] | | |
| costs differential misclassification costs [optional] | | |
| tree decision tree classifier produced by C5.0 [output] | | |
| rules ruleset classifier produced by C5.0 [output] | | |

### Names file

Two files are essential for all C5.0 applications and there are three
further optional files, each identified by its extension.
The first essential file is
the names file (e.g.
`hypothyroid.names`) that
describes the attributes and classes.
There are two important subgroups of attributes:

* The value of an *explicitly-defined attribute* is given directly
  in the data in one of several forms.
  A *discrete* attribute has a value drawn from
  a set of nominal values, a *continuous* attribute has a
  numeric value, a *date* attribute holds a calendar date,
  a *time* attribute holds a clock time,
  a *timestamp* attribute holds a date and time,
  and a *label* attribute serves only to identify a particular case.* The value of an *implicitly-defined attribute*
    is specified by a formula.

The file `hypothyroid.names`
looks like this:

```
diagnosis.                     | the target attribute

age:                           continuous.
sex:                           M, F.
on thyroxine:                  f, t.
query on thyroxine:            f, t.
on antithyroid medication:     f, t.
sick:                          f, t.
pregnant:                      f, t.
thyroid surgery:               f, t.
I131 treatment:                f, t.
query hypothyroid:             f, t.
query hyperthyroid:            f, t.
lithium:                       f, t.
tumor:                         f, t.
goitre:                        f, t.
hypopituitary:                 f, t.
psych:                         f, t.
TSH:                           continuous.
T3:                            continuous.
TT4:                           continuous.
T4U:                           continuous.
FTI:=                          TT4 / T4U.
referral source:               WEST, STMW, SVHC, SVI, SVHD, other.

diagnosis:                     primary, compensated, secondary, negative.

ID:                            label.
```

#### What's in a name?

Names, labels, classes, and discrete values are represented by arbitrary
strings of characters, with some fine print:

* Tabs and spaces are permitted inside a name or value, but C5.0
  collapses every sequence of these characters to a single space.* Special characters (comma, colon, period, vertical bar ``|`')
    can appear
    in names and values, but must be prefixed by the escape character
    ``\`'.
    For example, the name "Filch, Grabbit, and Co." would be written
    as ``Filch\, Grabbit\, and Co\.`'.
    (Colons in times and periods in numbers do not need to be escaped.)

Whitespace (blank lines, spaces, and tab characters) is ignored
except inside a name or value and can be used to improve legibility.
Unless it is escaped as above,
the vertical bar ``|`'
causes the remainder of the line to be ignored and is handy for
including comments.
This use of ``|`' should not occur inside a value.

#### Specifying the classes

The first entry in the names file specifies the classes
in one of three formats:

* A list of class names separated by commas, e.g.
  `primary, compensated, secondary, negative.`* The name of a discrete attribute (the *target* attribute)
    that contains the class value, e.g.:
    `diagnosis.`* The name of a continuous target attribute followed by a colon and
      one or more thresholds in increasing order and separated by commas.
      If there are t thresholds
      X1, X2, ..., Xt
      then the values of the attribute are divided into
      t+1 ranges:
      + less than or equal to X1+ greater than X1 and less than or equal to X2+ . . .+ greater than Xt.Each range defines a class, so there are t+1 classes. For example,
      a hypothetical entry
      `age: 12, 19.`would define three
      classes: `age <= 12`, `12 < age <= 19`, and
      `age > 19`.

This first entry defining the classes is followed by definitions of
the attributes in the order that they will be given for each case.

#### Explicitly-defined attributes

The name of each explicitly-defined attribute is followed by a colon
``:`' and a description of the values taken by the attribute.
The attribute name is arbitrary, except that each attribute must have
a distinct name, and `case weight`
is reserved for setting weights for individual cases.
There are eight possibilities for the description of attribute values:

`continuous`: The attribute takes numeric values. `date`: The attribute's values are dates in the form YYYY/MM/DD or YYYY-MM-DD, e.g. `2005/09/30` or `2005-09-30`. `time`: The attribute's values are times in the form HH:MM:SS with values between `00:00:00` and `23:59:59`. `timestamp`: The attribute's values are times in the form YYYY/MM/DD HH:MM:SS or YYYY-MM-DD HH:MM:SS, e.g. `2005-09-30 15:04:00`. (Note that there is a space separating the date and time.) a comma-separated list of names: The attribute takes discrete values, and these are the allowable values. The values may be prefaced by `[ordered]` to indicate that they are given in a meaningful ordering, otherwise they will be taken as unordered. For instance, the values `low, medium, high` are ordered, while `meat, poultry, fish, vegetables` are not. The former might be declared as ```` ``` grade: [ordered] low, medium, high. ``` ```` If the attribute values have a natural order, it is better to declare them as such so that C5.0 can exploit the ordering. (*NB:* The target attribute should not be declared as ordered.) `discrete` N for some integer N: The attribute has discrete, unordered values, but the values are assembled from the data itself; N is the maximum number of such values. This form can be handy for unordered discrete attributes with many values, but its use means that the data values cannot be checked. (*NB:* This form cannot be used for the target attribute.) `ignore`: The values of the attribute should be ignored. `label`: This attribute contains an identifying label for each case, such as an account number or an order code. The value of the attribute is ignored when classifiers are constructed, but is used when referring to individual cases. A label attribute can make it easier to locate errors in the data and to cross-reference results to individual cases. If there are two or more label attributes, only the last is used.

#### Attributes defined by formulas

The name of each implicitly-defined attribute is followed by
``:=`'
and then a formula defining the attribute value. The formula is
written in the usual way, using parentheses where needed, and
may refer to any attribute defined up to this point.
Constants in the formula can be
numbers written in decimal notation, dates, times,
and discrete attribute values enclosed in string quotes ``"`'.
The operators and functions that can be used in the formula are

* `+`, `-`, `*`,
  `/`, `%` (mod),
  `^` (meaning `raised to the power')* `>`, `>=`, `<`,
    `<=`, `=`, `<>` or
    `!=` (both meaning `not equal')* `and`, `or`* `sin(`...`)`,
        `cos(`...`)`,
        `tan(`...`)`,
        `log(`...`)`,
        `exp(`...`)`,
        `int(`...`)` (meaning `integer part of')

The value of such an attribute is either continuous or true/false
depending on the formula. For example, the attribute
```` ```
	FTI:=  TT4 / T4U.
``` ````
is continuous since its value is obtained by dividing one number by
another. The value of a hypothetical attribute such as
```` ```
	strange := referral source = "WEST" or age > 40.
``` ````
would be either `t` or `f`
since the value given by the formula is either true or false.

If the value of the formula cannot be determined for a particular
case because one or more of the attributes appearing in the formula
have unknown or non-applicable values,
the value of the implicitly-defined attribute is unknown.

#### Dates, times, and timestamps

Dates are stored by C5.0 as the number of days since a particular
starting point
so some operations on dates make sense.
Thus, if we have attributes
```` ```
	d1: date.
	d2: date.
``` ````
we could define
```` ```
	interval := d2 - d1.
	gap := d1 <= d2 - 7.
	d1-day-of-week := (d1 + 1) % 7 + 1.
``` ````
`interval` then represents the number of days from
`d1` to `d2` (non-inclusive) and
`gap` would have a true/false value signaling whether
`d1` is at least a week before `d2`.
The last definition is a slightly non-obvious way of determining
the day of the week on which `d1` falls, with values
ranging from 1 (Monday) to 7 (Sunday).

Similarly, times are stored as the number of seconds since midnight.
If the names file includes
```` ```
	start: time.
	finish: time.
	elapsed := finish - start.
``` ````
the value of `elapsed` is the number of seconds
from `start` to `finish`.

Timestamps are a little more complex. A timestamp is rounded to
the nearest minute, but limitations on the precision of floating-point
numbers mean that the values stored for timestamps from more than
thirty years ago are approximate.
If the names file includes
```` ```
	departure: timestamp.
	arrival: timestamp.
	flight time := arrival - departure.
``` ````
the value of `flight time` is the number of minutes
from `departure` to `arrival`.

#### Selecting the attributes that can appear in classifiers

An optional final entry in the names file affects the
way that C5.0 constructs classifiers.
This entry takes one of
the forms
```` ```
	attributes included:
	attributes excluded:
``` ````
followed by a comma-separated list of attribute names. The first
form restricts the attributes used in classifiers to those specifically
named;
the second form specifies that classifiers must not use any of the named
attributes.

Excluding an attribute from classifiers is not the same as ignoring the
attribute (see ``ignore`' above).
As an example, suppose that numeric attributes `A`
and `B`
are defined in the data, but background knowledge suggests that
only their difference is important.
The names file might then contain the following entries:

```
	. . .
	A: continuous.
	B: continuous.
	Diff := A - B.
	   . . .
	attributes excluded: A, B.
```

In this example the attributes `A` and `B`
could not be defined
as `ignore` because the definition of `Diff`
would then be invalid.

### Data file

The second essential file,
the application's data file
(e.g. `hypothyroid.data`)
provides information on the
*training*
cases from which C5.0 will extract patterns.
The entry for each case consists of one or more lines that give
the values for all explicitly-defined attributes. If the classes are listed
in the first line of the names file,
the attribute values are followed by the case's class value.
Values are separated by commas and the entry is optionally terminated by
a period.
Once again, anything on a line after a vertical bar ``|`'
is ignored.
(If the information for a case occupies more than one line, make sure
that the line breaks occur after commas.)

For example,
the first three cases from file
`hypothyroid.data` are:
```` ```
	41,F,f,f,f,f,f,f,f,f,f,f,f,f,f,f,1.3,2.5,125,1.14,SVHC,negative,3733
	23,F,f,f,f,f,f,f,f,f,f,f,f,f,f,f,4.1,2,102,?,other,negative,1442
	46,M,f,f,f,f,N/A,f,f,f,f,f,f,f,f,f,0.98,?,109,0.91,other,negative,2965
``` ````
Don't forget the commas between values!
If you leave them out,
C5.0 will not be able to process your data.

Notice that
``?`' is used to denote a value that is missing or unknown.
Similarly, ``N/A`' denotes a value that is not applicable for
a particular case.
Also note
that the cases do not contain values for the attribute `FTI`
since its values are computed from other attribute values.

### Test and cases files (optional)

Of course, the value of predictive patterns lies in their ability to make
accurate predictions!
It is difficult to judge the accuracy of a classifier by measuring
how well it does on the cases used in its construction;
the performance of the classifier on
new cases is much more informative.
(For instance, any number of gurus tell us about patterns that `explain'
the rise/fall behavior of the stock market in the past. Even though
these patterns may appear plausible, they are only valuable to the extent that
they make useful predictions about future rises and falls.)

The third kind of file used
by C5.0 consists of new test
cases (e.g. `hypothyroid.test`) on which the classifier
can be evaluated.
This file is optional and, if used, has
exactly the same format as the data file.

Another optional file, the cases file
(e.g. `hypothyroid.cases`),
differs from a test file only in allowing the cases'
classes to be unknown (``?`').
The cases file is used primarily with
the public source code
described later on under [linking to other programs](#HOOKS).

### Costs file (optional)

The last kind of file, the costs file
(e.g. `hypothyroid.costs`),
is also optional and sets out
differential misclassification costs.
In some applications
there is a much higher penalty for certain types of mistakes.
In this application, a prediction that hypothyroidism is not present
could be very costly if in fact it is.
On the other hand, predicting incorrectly that a patient is
hypothyroid
may be a less serious error.
C5.0 allows different misclassification
costs to be associated with each combination of real class and
predicted class. We will return to this topic near the end of the
tutorial.

## Constructing Classifiers

Once the names, data, and optional
files have been set up, everything is ready to use C5.0.

The general form of the Unix command is

```
        c5.0 -f filestem [options]
```

This invokes C5.0 with the `-f`
option that identifies the application name
(here `hypothyroid`).
If no filestem is specified using this option, C5.0 uses a default
filestem that is almost certainly incorrect.
(Moral: always use the `-f` option!)
There are several options that affect the type of classifier that
C5.0 produces and the way that it is constructed.
Many of the options have default values that should be satisfactory
for most applications.

### Decision trees

When C5.0 is invoked with
no options, as

```
	c5.0 -f hypothyroid
```

it constructs a decision tree and generates output
like this:

```
C5.0 [Release 2.11a]    Sun Apr  7 08:48:18 2019
--------------------

    Options:
        Application `hypothyroid'

Class specified by attribute `diagnosis'

Read 2772 cases (24 attributes) from hypothyroid.data

Decision tree:

TSH <= 6: negative (2472/2)
TSH > 6:
:...FTI <= 65.3:
    :...thyroid surgery = t: negative (4.2/2)
    :   thyroid surgery = f:
    :   :...TT4 <= 61: primary (51/3.7)
    :       TT4 > 61:
    :       :...referral source in {WEST,SVHD}: primary (0)
    :           referral source in {STMW,SVHC,SVI}: primary (4.9/0.8)
    :           referral source = other:
    :           :...TSH <= 22: negative (6.4/2.7)
    :               TSH > 22: primary (5.8/0.8)
    FTI > 65.3:
    :...on thyroxine = t: negative (37.7)
        on thyroxine = f:
        :...thyroid surgery = t: negative (6.8)
            thyroid surgery = f:
            :...TT4 > 153: negative (6/0.1)
                TT4 <= 153:
                :...TT4 <= 37: primary (2.5/0.2)
                    TT4 > 37: compensated (174.6/24.8)

Evaluation on training data (2772 cases):

            Decision Tree
          ----------------
          Size      Errors

            11    8( 0.3%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            58     3           2    (a): class primary
                 154                (b): class compensated
                               2    (c): class secondary
                   1        2552    (d): class negative

        Attribute usage:

             90%  TSH
             18%  thyroid surgery
             18%  on thyroxine
             14%  TT4
             13%  T4U
             13%  FTI
              7%  referral source

Evaluation on test data (1000 cases):

            Decision Tree
          ----------------
          Size      Errors

            11    3( 0.3%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            31     1                (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
                   1         927    (d): class negative

Time: 0.0 secs
```

The first part identifies the version of C5.0, the run date,
and the options with which the system was invoked.
C5.0 constructs a decision tree from the 2772 training cases
in the file `hypothyroid.data`, and this appears next.
Although it may not look much like a tree, this output can be
paraphrased as:

```
   if TSH is less than or equal to 6 then negative
        else
        if TSH is greater than 6 then
            if FTI is less than or equal to 65.3 then
	        if thyroid surgery is true then negative
		else
		if thyroid surgery is false then
		    if TT4 is less than or equal to 61 then primary
		    else
		    if TT4 is greater than 61 then
			. . . .
```

and so on.

**Please note:** This explanation oversimplifies threshold tests such as
`TSH <= 6`; the details are discussed later in the section
[Soft thresholds](#SOFT).

The tree employs a case's attribute values to map it
to a *leaf* designating one of the classes.
Every leaf of the tree is followed by a cryptic (n) or
(n/m).
For instance, the last leaf of the decision tree
is `compensated (174.6/24.8)`, for which n is 174.6 and
m is 24.8.
The value of n is the number of cases in the file
`hypothyroid.data`
that are mapped to this leaf, and m (if it appears) is the number of
them that are classified incorrectly by the leaf.
(A non-integral number of cases can arise because, when the value of
an attribute in the tree is not known, C5.0 splits the case
and sends a fraction down each branch.)

The next section covers the evaluation of this decision tree shown in the
second part of the output.
Before we leave this output, though, its final line
states the *elapsed* time for the run.
A decision tree is usually constructed quickly, even
when there are many thousands of cases. Some of the options described later,
such as ruleset generation and boosting, can slow things down considerably.

The progress of C5.0 on long runs can be monitored by examining the
last few lines of the temporary
file filestem`.tmp`
(e.g. `hypothyroid.tmp`).
This file displays the stage that C5.0 has reached and, for most stages,
gives an indication of progress within that stage.

### Evaluation

Classifiers constructed by C5.0 are evaluated on the training data
from which they were generated, and also on a separate file of
unseen test cases if available; evaluation by cross-validation is
discussed later.

Results of the decision tree on the cases in
`hypothyroid.data` are:

```
            Decision Tree
          ----------------
          Size      Errors

            11    8( 0.3%)   <<
```

`Size` is the number of non-empty leaves on the
tree and
`Errors` shows
the number and percentage of cases misclassified.
The tree, with 11 leaves, misclassifies 8 of the 2772 given cases, an error
rate of 0.3%.
This might seem inconsistent with the errors recorded at the leaves --
the leaf mentioned above shows 24.8 errors! The discrepancy arises because
parts of a case split as a result of unknown attribute values can
be misclassified and yet, when the votes from all the parts are aggregated,
the correct class can still be chosen.

When there are no more than twenty classes,
performance on the training cases is further analyzed in a
*confusion matrix*
that pinpoints the kinds of errors made.

```
           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            58     3           2    (a): class primary
                 154                (b): class compensated
                               2    (c): class secondary
                   1        2552    (d): class negative
```

In this example, the decision tree misclassifies

* three of the `primary` cases as
  `compensated`
  and two of them as `negative`,* both `secondary` cases as
    `negative`, and* one `negative` case as
      `compensated`.

When the number of classes is larger than twenty,
a summary of performance broken down by class is shown instead.
The entry for each class shows the number
of cases for that class and the numbers of *false positives* and *false
negatives*. A false positive for class C is a case of
another class that is classified as C, while a false
negative for C is a case of class C that is
classified as some other class. Of course, the total number of errors
must come to half the sum of the numbers of false positives and false
negatives, since each error is counted twice--as a false negative for
its true class, and as a false positive for the predicted class.

For some applications, especially those with many attributes, it
may be useful to know how the individual attributes contribute to
the classifier. This is shown in the next section:

```
        Attribute usage:

             90%  TSH
             18%  thyroid surgery
             18%  on thyroxine
             14%  TT4
             13%  T4U
             13%  FTI
              7%  referral source
```

The figure before each attribute is the percentage of training cases
in `hypothyroid.data` for which the value of that
attribute is known and is used in predicting a class. The second
entry, for instance, shows that the decision tree uses a known
value of `thyroid surgery` when classifying 18% of the
training cases. Attributes for which this value is less than 1%
are not shown. Two points are worth noting here:

* These values are computed for the particular classifier and training
  cases; changing either would give different values.* When a case is classified, use of an attribute such as
    `FTI` that is defined by a formula also counts as
    using any attributes involved in its definition (here `TT4`
    and `T4U`).

If there are optional unseen test cases, the
classifier's performance on these cases is summarized in a
format similar to that for the training cases.

```
            Decision Tree
          ----------------
          Size      Errors

            11    3( 0.3%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            31     1                (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
                   1         927    (d): class negative
```

A very simple *majority* classifier predicts
that every new case belongs to the most common class in the
training data.
In this example, 2553 of the 2772 training cases belong to class
`negative` so that
a majority classifier would always opt for
`negative`.
The 1000 test cases from file
`hypothyroid.test`
include 928 belonging to class `negative`,
so a simple
majority classifier would have an error rate of 7.2%.
The decision tree does much better than this with an error rate of 0.3%
on the test cases!

The error rate on the test cases is almost always higher than the
rate on the training cases.
The later section on cross-validation
discusses a more effective way of determining the predictive error
rate of a classifier.

The confusion matrix (or false positive/false negative summary if
there are more than twenty classes) for the test cases again provides
more details on correct and incorrect classifications.

### Discrete value subsets

By default, a test on
a discrete attributes has a separate branch for
each of its values that is present in the data.
Tests with a high fan-out can have the undesirable side-effect
of fragmenting the data during construction of the decision tree.
C5.0 has an option **`-s`** (lower-case *s*)
that can mitigate this fragmentation to some
extent: attribute values are grouped into subsets and each subtree
is associated with a subset rather than with a single value.

In the hypothyroid example, invoking this option
by the command

```
	c5.0 -f hypothyroid -s
```

as no effect on the tree produced.

Although it does not help for this application, the
**`-s`**
option is recommended when there are important discrete
attributes that have more than four or five values.

### Rulesets

Decision trees can sometimes be quite difficult to understand. An
important feature of C5.0 is its ability to generate
classifiers called *rulesets* that consist of unordered
collections of (relatively) simple if-then rules.

The option **`-r`** causes
classifiers to be expressed as rulesets rather than decision trees.
The command

```
	c5.0 -f hypothyroid -r
```

gives the following:

```
C5.0 [Release 2.11a]    Sun Apr  7 08:49:53 2019
--------------------

    Options:
        Application `hypothyroid'
        Rule-based classifiers

Class specified by attribute `diagnosis'

Read 2772 cases (24 attributes) from hypothyroid.data

Rules:

Rule 1: (31, lift 42.7)
        thyroid surgery = f
        TSH > 6
        TT4 <= 37
        ->  class primary  [0.970]

Rule 2: (63/6, lift 39.3)
        TSH > 6
        FTI <= 65.3
        ->  class primary  [0.892]

Rule 3: (270/116, lift 10.3)
        TSH > 6
        ->  class compensated  [0.570]

Rule 4: (2225/2, lift 1.1)
        TSH <= 6
        ->  class negative  [0.999]

Rule 5: (296, lift 1.1)
        on thyroxine = t
        FTI > 65.3
        ->  class negative  [0.997]

Rule 6: (240, lift 1.1)
        TT4 > 153
        ->  class negative  [0.996]

Rule 7: (29, lift 1.1)
        thyroid surgery = t
        FTI > 65.3
        ->  class negative  [0.968]

Default class: negative

Evaluation on training data (2772 cases):

                Rules
          ----------------
            No      Errors

             7   14( 0.5%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            60     3                (a): class primary
             1   153                (b): class compensated
                               2    (c): class secondary
             5     3        2545    (d): class negative

        Attribute usage:

             90%  TSH
             20%  TT4
             14%  T4U
             14%  FTI
             11%  on thyroxine
              2%  thyroid surgery

Evaluation on test data (1000 cases):

                Rules
          ----------------
            No      Errors

             7    5( 0.5%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            32                      (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
             1     3         924    (d): class negative

Time: 0.0 secs
```

Each rule consists of:

* A rule number -- this is quite arbitrary and serves only to identify
  the rule.* Statistics
    `(`n`, lift` x`)`
    or
    `(`n`/`m`,
    lift` x`)`
    that summarize the performance of the rule.
    Similarly to a leaf, n is the number of training cases covered
    by the rule and m, if it appears, shows how many of them
    do not belong to the class predicted by the rule.
    The rule's accuracy is estimated by the Laplace ratio
    (n-m+1)/(n+2).
    The lift x
    is the result of dividing the rule's estimated
    accuracy by the relative frequency of the
    predicted class in the training set.* One or more conditions that must all be satisfied if the rule
      is to be applicable.* A class predicted by the rule.* A value between 0 and 1 that indicates
          the confidence with which this prediction is made.
          (Note: The *boosting* option described below employs an
          artificial weighting of the training cases; if it is used,
          the confidence may not reflect the true accuracy of the rule.)

When a ruleset like this is used to classify a case, it may happen that
several of the rules are applicable (that is, all their
conditions are satisfied).
If the applicable rules predict different classes, there is
an implicit conflict that could be resolved in several ways:
for instance,
we could believe the rule with the highest confidence, or we could
attempt to aggregate the rules' predictions to reach a verdict.
C5.0 adopts the latter strategy --
each applicable rule votes for its predicted class
with a voting weight equal to its confidence value, the votes
are totted up, and
the class with the highest total vote is chosen as the final
prediction.
There is also a *default class*, here `negative`,
that is used when none of the rules apply.

Rulesets are generally easier to understand than trees
since each rule describes a specific context associated with
a class.
Furthermore,
a ruleset generated from a tree usually has fewer rules than
than the tree has leaves, another plus for comprehensibility.

Another advantage of ruleset classifiers is that they
can be more accurate predictors than decision
trees -- a point not illustrated here, since the ruleset has an
error rate of 0.5% on the test cases.
For very large datasets, however, generating rules with the
`-r` option
can require considerably more computer time.

For a given application, the attribute usage shown for a decision tree
and for a ruleset can be a bit different.
In the case of the tree, the attribute at the root is always
used (provided its value is known) while an attribute further down the
tree is used less frequently.
For a ruleset,
an attribute is used to classify a case if it is referenced by a
condition of at least one rule that applies to that case;
the order in which attributes appear in a ruleset is not relevant.

#### Rule utility ordering

The order of rules does not matter, so the default is to
group them by class and sub-ordered them by
confidence. An alternative ordering by estimated contribution to
predictive accuracy can be selected using the
`-u`
option. Under this option, the rule that most reduces the error rate
on the training data
appears first and the rule that contributes least appears last.
Furthermore, results are reported in a selected number of
*bands* so that the predictive accuracies of the more
important subsets of rules are also estimated. For example,
if the
option `-u 4`
is selected, the hypothyroid rules are reordered as

```
Rule 1: (2225/2, lift 1.1)
        TSH <= 6
        ->  class negative  [0.999]

Rule 2: (270/116, lift 10.3)
        TSH > 6
        ->  class compensated  [0.570]

Rule 3: (63/6, lift 39.3)
        TSH > 6
        FTI <= 65.3
        ->  class primary  [0.892]

Rule 4: (296, lift 1.1)
        on thyroxine = t
        FTI > 65.3
        ->  class negative  [0.997]

Rule 5: (240, lift 1.1)
        TT4 > 153
        ->  class negative  [0.996]

Rule 6: (29, lift 1.1)
        thyroid surgery = t
        FTI > 65.3
        ->  class negative  [0.968]

Rule 7: (31, lift 42.7)
        thyroid surgery = f
        TSH > 6
        TT4 <= 37
        ->  class primary  [0.970]
```

The rules are divided into four bands of roughly equal sizes
and a further summary is generated for both training and test cases.
Here is the output for test cases:

```
Evaluation on test data (1000 cases):

                Rules
          ----------------
            No      Errors

             7    5( 0.5%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            32                      (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
             1     3         924    (d): class negative

Rule utility summary:

        Rules         Errors
        -----         ------
        1-2        56( 5.6%)
        1-4        10( 1.0%)
        1-5         6( 0.6%)
```

This shows that,
when only the first two rules are used,
the error rate on the test cases is 5.6%,
dropping to 1.0% when the first four rules are used,
and so on. The performance of the entire ruleset is not
repeated since it is shown above the utility summary.

Rule utility orderings are not given for cross-validations (see below).

### Boosting

Another powerful feature incorporated in C5.0 is *adaptive boosting*, based
on the work of Rob Schapire and Yoav Freund.
The idea is to generate several classifiers (either decision trees
or rulesets) rather than just one. When a new case is to be
classified, each classifier votes for its predicted class and the votes
are counted to determine the final class.

But how can we generate several classifiers from a single dataset?
As the first step, a single decision tree or ruleset is constructed
as before from the training data (e.g. `hypothyroid.data`).
This classifier will usually make mistakes on some cases in the data;
the first decision tree, for instance, gives the wrong class
for 8 cases in `hypothyroid.data`.
When the second classifier is constructed, more attention is paid
to these cases in an attempt to get them right.
As a consequence, the second classifier will generally be different
from the first. It also will make errors on some cases,
and these become more important during construction
of the third classifier.
This process continues for a pre-determined number of iterations
or *trials*, but stops if the most recent classifiers is
either extremely accurate or too inaccurate.

The option **`-t` x** instructs C5.0 to
construct up to x
classifiers in this manner; an alternative option **`-b`**
is equivalent to `-t 10`.
Naturally, constructing multiple classifiers
requires more computation that building a single classifier
-- but the effort can pay dividends!
Trials over numerous datasets, large
and small, show that 10-classifier boosting reduces the
error rate for test cases by an average 25%.

In this example, the command

```
	c5.0 -f hypothyroid -b
```

causes ten decision trees to be generated.
The summary of the trees'
individual and aggregated performance on the 1000 test cases is:

```
Trial       Decision Tree
-----     ----------------
          Size      Errors

   0        11    3( 0.3%)
   1         5   63( 6.3%)
   2         8   17( 1.7%)
   3         9    6( 0.6%)
   4        12  119(11.9%)
   5        14   15( 1.5%)
   6         8   54( 5.4%)
   7        11   22( 2.2%)
   8        15   36( 3.6%)
   9        17   23( 2.3%)
boost             2( 0.2%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            32                      (a): class primary
                  40                (b): class compensated
                                    (c): class secondary
             1     1         926    (d): class negative
```

The performance of the classifier constructed at each trial
is summarized on a separate line, while the line labeled
`boost`
shows the result of voting all the classifiers.

The decision tree constructed on Trial 0 is identical to that
produced without the `-b` option.
Some of the subsequent trees produced by paying more attention
to certain cases
have relatively high overall error rates. Nevertheless, when the
trees are combined by voting,
the final predictions have a lower error rate of 0.2% on the test cases.

**Warning:**
An important characteristic of datasets is the extent to which they
are affected by *noise* -- incorrectly recorded values
of attributes or the class, or inherent probabilistic variability in
the classes themselves.
Boosting is particularly effective when the data are relatively noise-free
(such as the hypothyroid application), but can be counterproductive for
noisy datasets.

### Winnowing attributes

The decision trees and rulesets constructed by C5.0 do not generally
use all of the attributes. The hypothyroid application
has 22 predictive attributes (plus a class and a label attribute)
but only five of them appear in the tree and the ruleset.
This ability to pick and choose among the predictors is an
important advantage of tree-based modeling techniques.

Some applications, however, have an abundance of attributes!
For instance, one approach to text classification describes each
passage by the words that appear in it, so there is a separate attribute
for each different word in a restricted dictionary.

When there are numerous alternatives for each test in the tree or
ruleset, it is likely that at least one of them will appear to provide
valuable predictive information.
In applications like these it can be useful to pre-select a
subset of the attributes that will be used to construct the
decision tree or ruleset.
The C5.0 mechanism to do this is called "winnowing" by
analogy with the process for separating wheat from chaff (or, here,
useful attributes from unhelpful ones).

Winnowing is not obviously relevant for the hypothyroid application
since there are relatively few attributes. To illustrate the idea,
however, here are the results when the
**`-w`**
option is invoked:

```
C5.0 [Release 2.11a]    Sun Apr  7 08:52:49 2019
--------------------

    Options:
        Application `hypothyroid'
        Winnow attributes

Class specified by attribute `diagnosis'

Read 2772 cases (24 attributes) from hypothyroid.data

14 attributes winnowed
Estimated importance of remaining attributes:

    990%  TSH
    270%  FTI
    200%  on thyroxine
     30%  thyroid surgery
     <1%  age
     <1%  T3
     <1%  TT4
     <1%  referral source

Decision tree:

TSH <= 6: negative (2472/2)
TSH > 6:
:...FTI <= 65.3:
    :...thyroid surgery = t:
    :   :...FTI <= 38.2: negative (2.1)
    :   :   FTI > 38.2: primary (2.1/0.1)
    :   thyroid surgery = f:
    :   :...TT4 <= 61: primary (51/3.7)
    :       TT4 > 61:
    :       :...referral source in {WEST,SVHD}: primary (0)
    :           referral source in {STMW,SVHC,SVI}: primary (4.9/0.8)
    :           referral source = other:
    :           :...TSH <= 22: negative (6.4/2.7)
    :               TSH > 22: primary (5.8/0.8)
    FTI > 65.3:
    :...on thyroxine = t: negative (37.7)
        on thyroxine = f:
        :...thyroid surgery = t: negative (6.8)
            thyroid surgery = f:
            :...TT4 > 153: negative (6/0.1)
                TT4 <= 153:
                :...TT4 <= 37: primary (2.5/0.2)
                    TT4 > 37: compensated (174.6/24.8)

Evaluation on training data (2772 cases):

            Decision Tree
          ----------------
          Size      Errors

            12    6( 0.2%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            60     3                (a): class primary
                 154                (b): class compensated
                               2    (c): class secondary
                   1        2552    (d): class negative

        Attribute usage:

             90%  TSH
             18%  thyroid surgery
             18%  on thyroxine
             14%  TT4
             13%  T4U
             13%  FTI
              7%  referral source

Evaluation on test data (1000 cases):

            Decision Tree
          ----------------
          Size      Errors

            12    3( 0.3%)   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            31     1                (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
                   1         927    (d): class negative

Time: 0.0 secs
```

After analyzing the training cases and before the decision
tree is built, C5.0 winnows 14 of the 22 predictive attributes.
This has the same effect as marking the attributes as excluded
by an entry in the names file; winnowed attributes
can still be used in the definition of other attributes.
In this example, `T4U` is winnowed but is still available
for use in the definition of `FTI`.

The remaining attributes are then listed in order of *importance*,
C5.0's estimate of the factor by which the true error rate or
misclassification cost would increase if that attribute were excluded.
If TSH were excluded, for example, C5.0 expects the error rate on unseen
test cases to increase to 3% (990% of the current rate of 0.3%).
This estimate is only a rough guide
and should not be taken too literally!

We then see the decision tree that is constructed from the reduced
set of attributes. In this case it is
slightly more elaborate
than the original tree but has the same error rate on the test cases.

Since winnowing the attributes can be a time-consuming process,
it is recommended primarily for larger applications (100,000 cases or more)
where there is reason to suspect that
many of the attributes have at best marginal relevance to the
classification task.

### Soft thresholds

The top of our initial decision tree tests whether
the value of the attribute `TSH` is less than or
equal to, or greater than, 6. If the former holds, we go no further
and predict that the case's class is `negative`, otherwise
we perform further tests before making a decision.
Thresholds like this are sharp, so that a case with
a hypothetical value of 5.99 for `TSH` is treated
quite differently from one with a value of 6.01.

For some domains, this sudden change is quite appropriate --
for instance, there are hard-and-fast cutoffs for bands
of the income tax table. For other applications, though,
it is more reasonable to expect classification decisions to
change more slowly near the thresholds.

Each threshold in a decision tree actually consists of three parts --
a lower bound lb, an upper bound ub, and an
intermediate value t, the threshold shown in the
original decision tree. If the attribute value in question
is below lb or above ub, classification is
carried out using the single branch
corresponding to the `<=' or '>' result respectively.
If the value lies between lb and ub, *both*
branches of the tree are investigated and the results combined.
The relative weight of the '<=' branch (green) and the '>' branch (blue)
as the value of the attribute varies is shown in this figure:

![Soft threshold weights](threshold.png)
The values of lb and ub are determined by C5.0
based on an analysis of the apparent sensitivity of classification
to small changes in the threshold. They need not be symmetric --
a threshold can be sharper on one side than on the other.

C5.0 contains an option
**`-p`**
to display all the information about each threshold.
The command

```
	c5.0 -f hypothyroid -p
```

gives the following decision tree:

```
TSH <= 6 (6): negative (2472/2)
TSH >= 6.1 (6):
:...FTI <= 61 (65.3):
    :...thyroid surgery = t: negative (4.2/2)
    :   thyroid surgery = f:
    :   :...TT4 <= 44 (61): primary (51/3.7)
    :       TT4 >= 63 (61):
    :       :...referral source in {WEST,SVHD}: primary (0)
    :           referral source in {STMW,SVHC,SVI}: primary (4.9/0.8)
    :           referral source = other:
    :           :...TSH <= 14 (22): negative (6.4/2.7)
    :               TSH >= 44 (22): primary (5.8/0.8)
    FTI >= 66 (65.3):
    :...on thyroxine = t: negative (37.7)
        on thyroxine = f:
        :...thyroid surgery = t: negative (6.8)
            thyroid surgery = f:
            :...TT4 >= 159 (153): negative (6/0.1)
                TT4 <= 146 (153):
                :...TT4 <= 14 (37): primary (2.5/0.2)
                    TT4 >= 61 (37): compensated (174.6/24.8)
```

Each threshold is now of the form
`<=` lb `(`t`)`
or
`>=` ub `(`t`)`.
If a case has a value of the relevant attribute that is
between lb and t, both
branches are explored and the results combined with the relative weight of
the `<=` branch ranging from 1 to 0.5. Similarly, if a
case has a value between t and ub, the
relative weight of the `<=` branch ranges from 0.5 to 0.

A final point: soft thresholds affect only decision tree classifiers --
they do not change the interpretation of rulesets.

### Advanced pruning options

Three further options enable aspects of the classifier-generation
process to be tweaked. These are best regarded as *advanced*
options
that should be used sparingly (if at all), so that this
section can be skipped without much loss.

C5.0 constructs decision trees in two phases. A large tree is first
grown to fit the data closely and is then `pruned'
by removing parts that are predicted to have a relatively high error rate.
This pruning process is first applied to every subtree to decide
whether it should be replaced by a leaf or sub-branch, and then
a global stage looks at the performance of the tree as a whole.

The option **`-g`**
disables this second pruning component and generally results in
larger decision tees and rulesets. For the hypothyroid application,
the tree increases in size from 11 to 12 leaves.
(Turning off global pruning can be beneficial for smaller applications
with many discrete attributes.)

The option **`-c` CF**
affects the way that error rates are estimated
and hence the severity of pruning;
values smaller than the
default (25%) cause more of the initial tree to be pruned,
while larger values result in less pruning.

The option **`-m` cases**
constrains the degree to which the initial tree can fit the data.
At each branch point in the decision tree,
the stated minimum number of training cases must follow
at least two of the branches.
Values higher than the default (2 cases)
can lead to an initial tree that fits the training data only
approximately -- a form of pre-pruning.
(This option is complicated by the presence of missing attribute values,
and by the use of differential misclassification costs or weighting
of individual cases as discussed elsewhere.
All cause adjustments to the apparent number of cases following a branch.)

### Sampling from large datasets

Even though C5.0 is relatively fast, building classifiers from
large numbers of cases can take an inconveniently long time,
especially when options such as boosting are employed.
C5.0 incorporates a facility to extract a random sample from
a dataset, construct a classifier from the sample, and then test
the classifier on a disjoint collection of cases.
By using a smaller set of training cases in this way, the
process of generating a classifier is expedited,
but at the cost of a possible reduction in the classifier's
predictive performance.

The option **`-S` x**
(upper-case *S*)
has two consequences.
Firstly, a random sample containing x% of the cases in
the application's data file is used to construct the classifier.
Secondly, the classifier is evaluated on
a non-overlapping set of test cases consisting of
another (disjoint) sample of the same size as the training set
(if x is less than 50%),
or all cases that were not used in the training set
(if x is greater than or equal to 50%).

In the hypothyroid example,
using a sample of 60% would cause a classifier to be constructed
from a randomly-selected 1663 of the 2772 cases in
`hypothyroid.data`, then tested on the
remaining 1109 cases.

By default, the random sample changes every time that
a classifier is constructed, so that
successive runs of C5.0 with sampling will
usually produce different results.
This re-sampling can be avoided by the option
**`-I` seed**
that uses the integer seed to initialize the sampling.
Runs with the same value of the seed and the same sampling
percentage will always use the same training cases.

### Cross-validation trials

As we saw earlier, the performance of a classifier on the training
cases from which it was constructed gives a poor estimate of
its accuracy on new cases.
The true predictive accuracy of the classifier can be estimated
by sampling, as above, or by using a separate test file;
either way, the classifier is evaluated on cases that were
not used to build it.
However, this estimate can be unreliable unless the numbers of
cases used to build and evaluate the classifier are both large.
If the cases in `hypothyroid.data` and
`hypothyroid.test` were to be shuffled
and divided into a new 2772-case training set and a 1000-case test set,
C5.0 might construct a different classifier with a lower or higher error
rate on the test cases.

One way to get a more reliable estimate of predictive accuracy
is by *f-fold cross-validation*. The cases
(including those in the test file, if it exists)
are divided into
f blocks of roughly the same size and class distribution.
For each block in turn, a classifier is constructed from the
cases in the remaining blocks and tested on the cases in the
hold-out block. In this way, each case is used just once as
a test case. The error rate of a classifier produced from
all the cases is estimated as the ratio of the total number of errors
on the hold-out cases to the total number of cases.

The option **`-X` f**
runs such a f-fold cross-validation.
For example, the command

```
	c5.0 -f hypothyroid -X 10 -r
```

selects 10-fold cross-validation using rulesets.
After giving details of the individual rulesets,
the output shows a summary like this:

```
Fold            Rules
----      ----------------
            No      Errors

   1         7        0.5%
   2         7        0.3%
   3         7        0.5%
   4         8        0.5%
   5         8        1.6%
   6         7        0.5%
   7         7        0.0%
   8         7        0.0%
   9         7        0.3%
  10         7        0.8%

  Mean     7.2        0.5%
  SE       0.1        0.1%

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            91     4                (a): class primary
                 192           2    (b): class compensated
                               2    (c): class secondary
             5     6        3470    (d): class negative
```

This estimates the error rate of the rulesets
produced from the
3772 cases in `hypothyroid.data` and `hypothyroid.test`
at 0.5%.
The *SE* figures (the standard errors of the means)
provide an estimate of the variability of these results.

As with sampling above, each cross-validation run will normally use
a different random division of the data into blocks, unless this
is prevented by using the `-I` option.

The cross-validation procedure can be repeated for
different random partitions of the cases into blocks. The average
error rate from these distinct cross-validations is then an even more
reliable estimate of the error rate of the single classifier
produced from all the cases.

A shell script and associated programs for carrying out multiple
cross-validations is included with C5.0.
The shell script `xval` is invoked with any combination of C5.0
options and some further options that describe the cross-validations
themselves:

|  |  |
| --- | --- |
| `F=`folds | specifies the number of cross-validation folds (default 10) |
| `R=`repeats | causes the cross-validation to be repeated repeats times (default 1) |
| `+`suffix | adds the identifying suffix `+`suffix to all files |
| `+d` | retains the files output by individual runs |

If detailed results are retained via the `+d` option,
they appear in files named
filestem`.o`x[`+`suffix]
where x is the cross-validation number
(0 to repeats-1).
A summary of the cross-validations is written to file
filestem`.res`[`+`suffix].

As an example, the command

```
	xval -f hypothyroid -r R=10 +run1
```

has the effect of running ten 10-fold cross-validations
using ruleset classifiers (i.e., 100 classifiers in all).
File `hypothyroid.res+run1` contains the following summary:

```
	XVal          Rules
	----    ----------------
	            No    Errors

	  1        7.2      0.5%
	  2        7.2      0.6%
	  3        7.2      0.5%
	  4        7.2      0.6%
	  5        6.8      0.7%
	  6        7.0      0.5%
	  7        7.1      0.6%
	  8        7.3      0.6%
	  9        7.2      0.6%
	 10        7.2      0.6%

	Mean       7.1      0.6%
	SE         0.0      0.0%
```

Since every cross-validation fold produces a different
classifier using only part of the application's
data, **running a cross-validation does not cause a classifier to be
saved**.
To save a classifier for later use, simply
run C5.0 without employing cross-validation.

### Differential misclassification costs

Up to this point, all errors have been treated as equal -- we have
simply counted the number of errors made by a classifier to
summarize its performance.
Let us now turn to the situation in which the `cost' associated
with a classification error depends on the predicted and true class
of the misclassified case.

C5.0 allows costs to be assigned to any combination of predicted and
true class via entries in the optional file
filestem`.costs`.
Each entry has the form

```
	predicted class, true class: cost
```

where cost is any non-negative value.
The file may contain any number of entries;
if a particular combination is not specified explicitly, its
cost is taken to be 0 if the predicted class is correct and
1 otherwise.

To illustrate the idea, suppose that it was a much more serious
error to classify a hypothyroid patient as `negative`
than the converse.
A hypothetical costs file `hypothyroid.costs`
might look like this:

```
negative, primary: 7.5
negative, secondary: 5
negative, compensated: 5
```

This specifies that the cost of misclassifying any
`primary` patient as `negative` is 7.5,
and any
`secondary`, or
`compensated`
patient as `negative` is 5.
Since they are not given explicitly, all other errors
have cost 1.
In other words, failing to identify a sick patient is more costly.

A costs file is automatically read by C5.0 unless the
system is told to ignore it.
(The option **`-e`**
causes any costs file to be ignored and instructs C5.0
to focus only on errors.)
The command

```
	c5.0 -f hypothyroid
```

now gives the following output:

```
C5.0 [Release 2.11a]    Sun Apr  7 08:56:12 2019
--------------------

    Options:
        Application `hypothyroid'

Class specified by attribute `diagnosis'

Read 2772 cases (24 attributes) from hypothyroid.data
Read misclassification costs from hypothyroid.costs

Decision tree:

TSH <= 6:
:...TT4 > 55: negative (2444.3)
:   TT4 <= 55:
:   :...query hypothyroid = f: negative (25.7/1)
:       query hypothyroid = t: secondary (2.1/1.1)
TSH > 6:
:...FTI <= 65.3:
    :...thyroid surgery = f: primary (68.1/11.7)
    :   thyroid surgery = t:
    :   :...FTI <= 38.2: negative (2.1)
    :       FTI > 38.2: primary (2.1/0.1)
    FTI > 65.3:
    :...on thyroxine = t: negative (37.7)
        on thyroxine = f:
        :...thyroid surgery = t: negative (6.8)
            thyroid surgery = f:
            :...TT4 > 153: negative (6/0.1)
                TT4 <= 153:
                :...TT4 <= 37: primary (2.5/0.2)
                    TT4 > 37: compensated (174.6/24.8)

Evaluation on training data (2772 cases):

               Decision Tree
          -----------------------
          Size      Errors   Cost

            11   12( 0.4%)   0.01   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            60     3                (a): class primary
             1   153                (b): class compensated
                         1     1    (c): class secondary
             5     1     1  2546    (d): class negative

        Attribute usage:

             94%  TT4
             90%  TSH
             18%  thyroid surgery
             18%  on thyroxine
             13%  T4U
             13%  FTI
              8%  query hypothyroid

Evaluation on test data (1000 cases):

               Decision Tree
          -----------------------
          Size      Errors   Cost

            11    4( 0.4%)   0.00   <<

           (a)   (b)   (c)   (d)    <-classified as
          ----  ----  ----  ----
            32                      (a): class primary
             1    39                (b): class compensated
                                    (c): class secondary
             1     2         925    (d): class negative

Time: 0.0 secs
```

The original tree and new tree are quite similar, but we can
compare the total cost of misclassified
cases for the two trees.
For the training cases, the original tree had cost 29
(two `primary` and two `secondary`
cases are classified as `negative` and there are
four other errors, so 2\*7.5 + 2\*5 + 4). The new tree
classifies one `secondary` case as `negative`
and makes 11 other errors for a total cost of 16.
Unfortunately, the new tree makes one additional error on the test cases
and so is of no benefit in this example.

The new "Cost" column in the output shows the average misclassification
cost, i.e. the total cost divided by the number of cases.
For the new tree, the average cost is 16/2772 for the training cases
and 4/1000 for the test cases.

### Weighting individual cases

It is sometimes useful to attach different weights to cases
depending on some measure of their importance.
An application predicting whether a customer is likely to "churn," for
example, might weight training cases by the size of the account.

C5.0 accommodates this by allowing a special attribute that contains
the weight of each case. The attribute name must be
`case weight` and
it must be of type `continuous`. The relative weight
assigned to each case is its value of this attribute divided by
the average value; if the value is undefined ("`?`"),
not applicable ("`N/A`"), or is less than or equal to zero,
the case's relative weight is set to 1.

*The case weight attribute itself is not used in the classifier!*

Our sample hypothyroid application does not have any natural case-by-case
weighting, since all patients are equal. For the purpose of illustration,
though, we will add an implicitly-defined attribute to
`hypothyroid.names` as follows:

`case weight := 100-age.`

With all default options, C5.0 now generates a different decision tree:

```
TSH <= 6: negative (2462.5/2.3)
TSH > 6:
:...FTI <= 64: primary (69.6/12.8)
    FTI > 64:
    :...on thyroxine = t: negative (39.6)
        on thyroxine = f:
        :...thyroid surgery = t: negative (9.2)
            thyroid surgery = f:
            :...TT4 > 153: negative (6.2/0.2)
                TT4 <= 153:
                :...TT4 > 61: compensated (179/29.3)
                    TT4 <= 61:
                    :...TSH <= 35: compensated (2.8/0.3)
                        TSH > 35: primary (3.2/0.3)
```

The case counts at the leaves now reflect the relative weights
of the cases. (The counts associated with rules are affected similarly.)
However, the error counts, rates, and costs shown in the evaluations
use uniform case weighting.

A cautionary note: The use of case weighting does not guarantee that
the classifier will be more accurate for unseen cases with higher
weights. Predictive accuracy on more important cases is likely
to be improved only when cases with similar values of the predictor
attributes also have similar values of the case weight attribute,
i.e. when relatively important cases "clump together."
Without this property, case weighting can introduce an unhelpful
element of randomness into the classifier generation process.

## Using Classifiers

Once a classifier has been constructed, an *interactive interpreter*
can be used to
predict the classes to which new cases belong.
The command to do this is

```
	predict
```

whose options are:

|  |  |
| --- | --- |
| `-f` filestem | to identify the application |
| `-r` | uses ruleset classifiers rather than decision trees |
| `-p` | prints the classifier as a reminder |

This is illustrated in the following dialog that uses the first decision
tree to predict the class of a case. Input from the user
is shown in bold face and the enter key as
**`¤`**.

```
TSH: 7.4¤
TT4: 108¤
T4U: 1.08¤
on thyroxine: f¤
thyroid surgery: f¤

    ->	compensated  [0.85]
	negative  [0.13]
	primary  [0.01]

Retry, new case or quit [r,n,q]: r¤

TSH [7.4]: ¤
TT4 [108]: ¤
T4U [1.08]: ¤
on thyroxine [f]: t¤

    ->	negative  [1.00]

Retry, new case or quit [r,n,q]: q¤
```

The values of some attributes might not affect the classification, so
`predict` prompts for the values of those attributes that
are required. The reply ``?`' indicates
that a requested attribute value is unknown.
(Similarly, use ``N/A`' for non-applicable values.)
When all the relevant information has been entered, the most likely
class (or classes) are printed, each with a confidence value.

Next, `predict` asks whether the same case is to be tried
again with changed attribute values (a kind of `what if'
scenario), a new case is to be classified, or all cases are complete.
If a case is retried, each prompt for an attribute value shows
the previous value in square brackets.
A new value can be entered, followed by the enter key, or
the enter key alone can be used to indicate that the value is unchanged.

Classifiers can also be used in batch mode.
The sample application provided in the
[public source code](http://rulequest.com/see5-win.html#HOOKS)
reads cases from a cases file and shows
the predicted class and the confidence for each.

## Segmentation fault errors

For applications with very many cases or attributes, C5.0 may crash
with a message like `Segmentation fault (core dumped)`. This
usually occurs because a C5.0 thread has exhausted its allocated stack
space.

Different releases of Linux have varying default stack sizes; for
example, Fedora Core uses a default 10MB while Ubuntu uses 8MB.
If you experience a segmentation fault error, you must
override the default stack size setting by typing the following
*before* running C5.0:

* if you are using *csh*:
  `limit stacksize` new limit* if you are using *sh*:
     `ulimit -Ss` new limit

where new limit is the new stack size limit in kilobytes.
For example, a csh user might type `limit stacksize 20000`
to increase the default stack size limit to 20MB.

**Please note!** You should not set the stack size limit to
`unlimited` -- this will not change the default stack size
limit for subsidiary threads. You must use a specific value in KB.
A little experimentation may be necessary to find a value that works
with your application.

## Linux GUI

Linux users who have installed a recent version of
[Wine](http://www.winehq.org) can invoke a
slightly simplified version of the See5 user interface.
The executable program `gui` starts the graphical
user interface whose main window is similar to See5's, with
five buttons:

Locate Data: invokes a browser to find the files for your application, or to change the current application; Construct Classifier: selects the type of classifier to be constructed and sets other options; Stop: interrupts the classifier-generating process; Review Output: re-displays the output from the last classifier construction (if any), saved automatically in a file filestem`.out`; and Cross-Reference: shows how cases in training or test data relate to (parts of) a classifier and vice versa.

For more details on these, please see the
[See5 tutorial](http://rulequest.com/see5-win.html).

The graphical interface calls C5.0 directly, so use of the GUI
has minimal impact on performance when generating a classifier.

**Please note:** C5.0 should be run for the first time from the
command-line interface, not the GUI. The first run installs the
licence in C5.0 -- after that, C5.0 can be used from either interface.

## Linking to Other Programs

The classifiers generated by C5.0 are retained in files
filestem`.tree` (for decision trees) and
filestem`.rules` (for rulesets).
Free C source code is available
to read these classifier files and to make predictions with them,
enabling you to use C5.0 classifiers in other
programs.

As an example, the source includes a program `sample.c` to input new cases
and to show how each is classified
by boosted or single trees or rulesets.
The program reads the application's names file, the
tree or rules file generated by C5.0,
and an optional costs file. It then reads cases from
a cases file in a format similar to a data
file, except that a case's class can be given as ``?`' meaning
"unknown". For each case, the program outputs the given class, the class
predicted by the classifier, and the confidence with which this prediction
is made.

Please see the file `sample.c` for compilation
instructions and program options.

[**Click here**](http://www.rulequest.com/see5-public.tgz)
to download a gzipped tar file
containing the public source code.

## Appendix: Summary of Options

|  |  |
| --- | --- |
| `-f` filestem select the application | |
| `-s` partition discrete values into subsets | |
| `-r` generate rule-based classifiers | |
| `-u` bands sort rules by their utility into bands | |
| `-b` use boosting with 10 trials | |
| `-t` trials use boosting with the specified number of trials | |
| `-w` winnow the attributes before constructing a classifier | |
| `-p` show soft thresholds | |
| `-g` do not use global tree pruning | |
| `-c` CF set the CF value for pruning trees | |
| `-m` cases set the minimum cases for at least two branches of a split | |
| `-S` x use a sample of x% for training and a disjoint sample for testing | |
| `-I` seed set the sampling seed value | |
| `-X` folds carry out a cross-validation | |
| `-e` ignore any costs file | |
| `-h` print a short summary of the options | |

|  |  |
| --- | --- |
| **© RULEQUEST RESEARCH 2019** | Last updated April 2019 |

---

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| [**home**](http://rulequest.com/index.html) [**products**](http://rulequest.com/products.html) [**download**](http://rulequest.com/download.html) [**evaluations**](http://rulequest.com/eval.html) [**prices**](http://rulequest.com/price.html) [**purchase**](http://rulequest.com/purchase.html) [**contact us**](http://rulequest.com/contact.html) | | | | | | |