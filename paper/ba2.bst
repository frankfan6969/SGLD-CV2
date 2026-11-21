%% This is file `ba2.bst'
%%
%% Please submit bugs or your comments to latex-support@vtex.lt
%% Start subject with a "[BA] bibtex: " + subject
%%
%% Sorting entries is done by name and year fields!
%%
%% You are free to use this style file as you see fit, provided 
%% that you do not make changes to the file. 
%% If you DO make changes, you are required to rename this file.
%%
%% It may be distributed under the terms of the LaTeX Project Public
%% License, as described in lppl.txt in the base LaTeX distribution.
%% Either version 1.0 or, at your option, any later version.
ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
    doi
  }
  {}
  { label extra.label sort.label short.list }
INTEGERS { output.state before.all mid.sentence after.sentence
           after.block after.authors between.elements bother slen set.settings default.info middle.edt tmp no.dash int str.length count return done }
           
FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #5 'between.elements :=
}
STRINGS { s t element}
STRINGS { longest.label last.label list.string temp.str tmp1}
STRINGS { v l f j b }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ newline$}
    { output.state after.block =
        { add.period$ write$
          newline$
         % "\newblock " write$
         "" write$
        }
        { output.state between.elements =
            { " " * write$ newline$}
            { output.state before.all =
                'write$
                { add.period$ " " * write$ newline$}
              if$
            }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {string.in.url}{
      'tmp1 :=
      url 't :=
      #0 'return :=
      #0 'done :=
      tmp1 text.length$ 'str.length :=
        { done not
          t empty$ not
          and
        }
        { t #1 str.length substring$ tmp1 =
            { #1 'return := 
              #1 'done := 
            }
            { t text.length$ str.length = 
                {#1 'done := }
                {t #2 global.max$ substring$ 't := }
              if$
            }
          if$
        }
      while$
      return 

}


FUNCTION {doi.in.url}{
  doi empty$
    {#0}
    {doi string.in.url}
  if$
}

FUNCTION {format.url}
{ url empty$
    'skip$
    {doi.in.url #1 =
       'skip$
       {"\url{" url * "}" * output }
     if$   
    } 
  if$
}

FUNCTION {strip.doi.doimain}{
 't := % content
  t text.length$ #10 >
    {
      t #1 #16 substring$ "https://doi.org/" = 
        {
          t #17 global.max$ substring$ 't :=
        }
        {
          t #1 #18 substring$ "http://dx.doi.org/" = 
            {t #19 global.max$ substring$ 't :=}
            'skip$
          if$
        }
      if$ 
    }
    'skip$
  if$ 
  t 
}

FUNCTION {format.doi}
{ 
  doi empty$
    'skip$
    {"\bid{doi={" doi strip.doi.doimain * "}}" * output}
  if$  
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {nocomma}
{ output.state before.all =
    'skip$
    { between.elements 'output.state := }
  if$
}

FUNCTION {add.blank}
{  " " * before.all 'output.state :=
}

FUNCTION {date.block}
{
  new.block
}

FUNCTION {fin.entry}
{ 
  format.url 
  format.doi   
  doi empty$
    'add.period$
    'skip$
  if$
  write$
  newline$
}

FUNCTION {make.tag}
{ 't :=
  duplicate$ empty$
    { pop$ "" }
    { bother #0 =
      { skip$ 
      }
      { "\b" t * "{" * swap$ * "}" * }
    if$
    }
  if$
}

FUNCTION {item.finish}
{
  fin.entry
}


FUNCTION {insert.element}
{ #0 bother =
      {element "{botherref}" * write$ newline$}
      {
         type$ "article" =
            {element "{barticle}" * write$ newline$}
            'skip$
         if$
         type$ "book" =
         type$ "proceedings" =
         type$ "manual" =
         type$ "booklet" =
         or or or 
            {element "{bbook}" * write$ newline$}
            'skip$
         if$
         type$ "inbook" =
            { "chapter_is_text" last.label =
                {
                  author empty$
                    {element "{bbook}" * write$ newline$}
                    {element "{bchapter}" * write$ newline$}
                  if$ 
                }
                { author empty$ editor empty$ or
                    {element "{bbook}" * write$ newline$}
                    {element "{bchapter}" * write$ newline$}
                  if$ 
                }
              if$ 
            }
            'skip$
         if$

     type$ "incollection" =
     type$ "inproceedings" =
     type$ "conference" =
     or or
            {element "{bchapter}" * write$ newline$}
            'skip$
         if$

      }
  if$
}
FUNCTION {end.element}
{ "\end" 'element :=
  item.finish
  insert.element
  "\endbibitem" write$ newline$
}
FUNCTION {begin.element}
{ "\begin" 'element :=
  insert.element
  #0 'middle.edt :=
}


STRINGS {z}

FUNCTION {remove.dots}
{ 'z :=
   ""
   { z empty$ not }
   { z #1 #2 substring$
     duplicate$ "\." =
       { z #3 global.max$ substring$ 'z :=  * }
       { pop$
         z #1 #1 substring$
         z #2 global.max$ substring$ 'z :=
         duplicate$ "." = 'pop$
           { * }
         if$
       }
     if$
   }
   while$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\emph{" swap$ * "}" * }
  if$
}
FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$
}

FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }

FUNCTION {space.word}
{ " " swap$ * " " * }
 % Here are the language-specific definitions for explicit words.
 % Each function has a name bbl.xxx where xxx is the English word.
 % The language selected here is ENGLISH
FUNCTION {bbl.and}
{ "and"}

FUNCTION {bbl.etal}
{ "et~al." }

FUNCTION {bbl.etal2}
{ "\betal\ " }


FUNCTION {bbl.editors}
{ "eds." }

FUNCTION {bbl.editor}
{ "ed." }

FUNCTION {bbl.edby}
{ "edited by" }

FUNCTION {bbl.edition}
{ "edition" }

FUNCTION {bbl.volume}
{ "volume" }

FUNCTION {bbl.of}
{ "of" }

FUNCTION {bbl.number}
{ "number" }

FUNCTION {bbl.nr}
{ "no." }

FUNCTION {bbl.in}
{ "in" }

FUNCTION {bbl.pages}
{ "" }

FUNCTION {bbl.page}
{ "" }

FUNCTION {bbl.chapter}
{ "chapter" }

FUNCTION {bbl.techrep}
{ "Technical Report" }

FUNCTION {bbl.mthesis}
{ "Master's thesis" }

FUNCTION {bbl.phdthesis}
{ "Ph.D. thesis" }

MACRO {jan} {"January"}

MACRO {feb} {"February"}

MACRO {mar} {"March"}

MACRO {apr} {"April"}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {oct} {"October"}

MACRO {nov} {"November"}

MACRO {dec} {"December"}

MACRO {acmcs} {"ACM Computing Surveys"}

MACRO {acta} {"Acta Informatica"}

MACRO {cacm} {"Communications of the ACM"}

MACRO {ibmjrd} {"IBM Journal of Research and Development"}

MACRO {ibmsj} {"IBM Systems Journal"}

MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

MACRO {ieeetc} {"IEEE Transactions on Computers"}

MACRO {ieeetcad}
 {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

MACRO {ipl} {"Information Processing Letters"}

MACRO {jacm} {"Journal of the ACM"}

MACRO {jcss} {"Journal of Computer and System Sciences"}

MACRO {scp} {"Science of Computer Programming"}

MACRO {sicomp} {"SIAM Journal on Computing"}

MACRO {tocs} {"ACM Transactions on Computer Systems"}

MACRO {tods} {"ACM Transactions on Database Systems"}

MACRO {tog} {"ACM Transactions on Graphics"}

MACRO {toms} {"ACM Transactions on Mathematical Software"}

MACRO {toois} {"ACM Transactions on Office Information Systems"}

MACRO {toplas} {"ACM Transactions on Programming Languages and Systems"}

MACRO {tcs} {"Theoretical Computer Science"}
FUNCTION {bibinfo.check}
{ swap$
  duplicate$ missing$
    {
      pop$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ pop$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    {
      swap$ "missing " swap$ * " in " * cite$ * warning$ pop$
      ""
    }
    { duplicate$ empty$
        {
          swap$ "empty " swap$ * " in " * cite$ * warning$
        }
        { swap$
          pop$
        }
      if$
    }
  if$
}
FUNCTION {write.url}
{ url empty$
    { skip$ }
    { "\newline\urlprefix\url{" url * "}" * write$ newline$ }
  if$
}

STRINGS  { bibinfo}
INTEGERS { nameptr namesleft numnames trunc}

FUNCTION {mk.tag}
{ 'temp.str :=
  duplicate$ empty$
    { pop$ "" }
    { 
      b "nothing" =
        'skip$
      {"\b" temp.str * "{" * swap$ * "}" * }
    if$
  }
  if$
}


FUNCTION {fmt.front.names}{
      l empty$
        'skip$
        { l *}    
      if$     

      f empty$
        'skip$
        {", " * f *}    
      if$       
      
      j empty$
        'skip$
        {" "  * j *}     
      if$
      
}

FUNCTION {name.separator}{
  namesleft #1 >
    {", " *}
    'skip$
  if$
}


FUNCTION {format.names}
{ 'b :=
  duplicate$ empty$ 'skip$ {
  's :=
  "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=

  s numnames "{ll}" format.name$ 'l :=
  l  "others" =
  l  "{others}" =
  l  "et al." =
  l  "{et al.}" =
  or or or
    {#1 'tmp :=}
    {#0 'tmp :=}
  if$

  numnames #6 >
    { #7 'namesleft :=
      #1 'trunc :=
    }
    { #0 'trunc := }
  if$

    { namesleft #0 > }
    { 

      s nameptr "{vv~}{ll}" format.name$  "snm"      mk.tag  'l :=   
      s nameptr "{f.}"      format.name$  "inits"    mk.tag  'f := 
      s nameptr "{jj}"      format.name$  "suffix"   mk.tag  'j := 
      namesleft #1 =
      tmp #1 = 
      and
        { 
          "\betal" 
          #1 'tmp :=
        }
        {
          b "bauthor" =
          b "beditor" =
          or
            { "\oauthor{" }
            {
              b "nothing" =
                {"" }
                {"\b" b * "{" *}
              if$
            }
          if$     

          fmt.front.names 

          b "nothing" =
            {""}
            {"}"}  
          if$ 
          *
        } 
      if$
       't :=

      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              numnames #2 >
                { "," * }
                'skip$
              if$
              trunc #1 =
                {
                  " " * bbl.etal2 *
                }
                {
                  s nameptr "{ll}" format.name$ duplicate$ "others" =
                    { 't := }
                    { pop$ }
                  if$
                  t "others" =
                    {
                      " " * bbl.etal2 *
                    }
                    {
                      bbl.and
                      space.word * t *
                    }
                  if$
                }
              if$
            }
          if$
        }
        't
      if$      
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
  } if$
} 

FUNCTION {format.key}
{ empty$
    { key field.or.null }
    { "" }
  if$
}

FUNCTION {format.authors}
{ author 
  bother #0 =
    {"bauthor"}
    {"author"}
  if$
  format.names}

FUNCTION {get.bbl.editor}
{ editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }

FUNCTION {format.editors}
{ 
  editor empty$
    'skip$
    {
      editor
      author empty$
        {"editor"}
        { bother #0 =
            {"nothing"}
            {"editor"}
          if$
        }
      if$  
      format.names

      " " *
      get.bbl.editor
   "(" swap$ * ")" *
      *
    }
  if$
}
FUNCTION {format.note}
{
 note empty$
    { "" }
    { note "comment" make.tag }
  if$
}

FUNCTION {format.atitle}
{ title 
  duplicate$ empty$ 'skip$
    { "atitle" make.tag}
  if$
  "title" bibinfo.check
}

FUNCTION {format.title}
{ title
  "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      %"\enquote{" swap$ *
      add.period$ 
      %"}" *
    }
  if$
}

FUNCTION {format.ctitle}
{ title 
  duplicate$ empty$ 'skip$
    { "ctitle" make.tag }
  if$
  "title" bibinfo.check
}

FUNCTION {format.full.names}
{'s :=
 "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{vv~}{ll}" format.name$
      't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                {
                  " " * bbl.etal *
                }
                {
                  numnames #2 >
                    { "," * }
                    'skip$
                  if$
                  bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {author.editor.key.full}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            'key
          if$
        }
        { editor format.full.names }
      if$
    }
    { author format.full.names }
  if$
}

FUNCTION {author.key.full}
{ author empty$
    { key empty$
         { cite$ #1 #3 substring$ }
          'key
      if$
    }
    { author format.full.names }
  if$
}

FUNCTION {editor.key.full}
{ editor empty$
    { key empty$
         { cite$ #1 #3 substring$ }
          'key
      if$
    }
    { editor format.full.names }
  if$
}

FUNCTION {make.full.names}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.full
    { type$ "proceedings" =
        'editor.key.full
        'author.key.full
      if$
    }
  if$
}

FUNCTION {label.year}
{year empty$
    {""}
    {year extra.label *
    }
  if$ 
}

FUNCTION {output.bibitem}
{ newline$
      "\bibitem[\protect\citeauthoryear{" 
          short.list *
      "}{" * 
          label.year * 
      "}]{" * 
          cite$ *
      "}" *
      write$
      newline$
      ""
  before.all 'output.state :=
}

FUNCTION {word.in}
{ bbl.in capitalize
  " " * }

FUNCTION {format.date}
{ year "year" make.tag "year" bibinfo.check duplicate$ empty$
    {
      "empty year in " cite$ * "; set to ????" * warning$
       pop$ "????"
    }
    'skip$
  if$
  extra.label *
  before.all 'output.state :=
  " (" swap$ * ")" *
}

FUNCTION {format.btitle}
{ title "btitle" make.tag "title" bibinfo.check
  duplicate$ empty$ 'skip$
    {
      emphasize
    }
  if$
}
FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume "seriesno" make.tag tie.or.space.prefix
      "volume" bibinfo.check * *
      series "sertitle" make.tag "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { swap$ bbl.of space.word * swap$
          emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series "sertitle" make.tag field.or.null }
        { series empty$
            { number "seriesno" make.tag "number" bibinfo.check }
            { output.state mid.sentence =
                { bbl.number }
                { bbl.number capitalize }
              if$
              number "seriesno" make.tag tie.or.space.prefix "number" bibinfo.check * *
              bbl.in space.word *
              series "sertitle" make.tag "series" bibinfo.check *
            }
          if$
        }
      if$
    }
    { "" }
  if$
}

FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    {
      output.state mid.sentence =
        { "l" }
        { "t" }
      if$ change.case$
      "edition" make.tag  
      "edition" bibinfo.check
      " " * bbl.edition *
    }
  if$
}
INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
  #0 'no.dash := 
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ "+" =
      or 
  { 
    #1 'multiresult := 
     t #1 #1 substring$ "-" =
        {#1 'no.dash := }
        'skip$
     if$    
  }
  { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}


FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}

FUNCTION {find.integer}
{ 't :=
  #0 'int :=
    { int not
      t empty$ not
      and
    }
    { t #1 #1 substring$ 's :=
      s is.num 
        {#1 'int :=}
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  int 
}


FUNCTION {do.pages}
{
  't :=
  "" 'j :=
  "" 'v :=
    { t empty$ not }
    { t #1 #1 substring$  's :=
        s "-" =
        { t #1 #2 substring$ "--" = not
            { t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { t #2 global.max$ substring$ 't := }
              while$
            }
          if$
          t 'v :=
          "" 't :=            
        }
        { 
          j  s * 'j :=
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
  j  "fpage" make.tag "--"  * v "lpage" make.tag *
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check 
        { pages 
          no.dash #0 =
            {"fpage" make.tag }
            {do.pages }
          if$
        }
        { pages  "fpage" make.tag }

      if$
    }
  if$
}

FUNCTION {format.journal.pages}
{ pages duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$
        { pop$ pop$ format.pages }
        {
          ", " *
          swap$
          pop$
          format.pages
          "pages" bibinfo.check
          *
        }
      if$
    }
  if$
}

FUNCTION {format.vol.num.pages}
{
 volume empty$ 
    {""}
    {
      nocomma
      volume "volume" make.tag 
    }
  if$
  format.journal.pages
}
FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check
        }
      if$
      chapter "comment" make.tag tie.or.space.prefix
      "chapter" bibinfo.check
      * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.booktitle}
{
  booktitle "btitle" make.tag "booktitle" bibinfo.check
  emphasize
}

FUNCTION {format.in.ed.booktitle}
{ format.booktitle duplicate$ empty$ 'skip$
    {
      editor empty$ 
        'skip$
        { editor
            author empty$
              {"editor"}
              { 
                bother #0 =
                  {"nothing"}
                  {"editor"}
                if$
              }
            if$  
          format.names
          get.bbl.editor
          " (" swap$ * "), " *
          * swap$
          * }
      if$
      word.in swap$ *
    }
  if$
}

FUNCTION {format.thesis.type}
{ type duplicate$ empty$
    'pop$
    { swap$ pop$
      "t" change.case$ "type" bibinfo.check
    }
  if$
}

FUNCTION {format.tr.number}
{ number "number" bibinfo.check
  type duplicate$ empty$
    { pop$ bbl.techrep }
    'skip$
  if$
  "type" bibinfo.check
  swap$ duplicate$ empty$
    { pop$ "t" change.case$ }
    { tie.or.space.prefix * * }
  if$
}
FUNCTION {format.article.crossref}
{
  word.in
  " \cite{" * crossref * "}" *
}
FUNCTION {format.book.crossref}
{ volume duplicate$ empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      pop$ word.in
    }
    { bbl.volume
      capitalize
      swap$ tie.or.space.prefix "volume" bibinfo.check * * bbl.of space.word *
    }
  if$
  " \cite{" * crossref * "}" *
}
FUNCTION {format.incoll.inproc.crossref}
{
  word.in
  " \cite{" * crossref * "}" *
}
FUNCTION {format.org.or.pub}
{ 's :=
  ""
  address empty$ s empty$ and
    'skip$
    {
      address empty$
        'skip$
        {address "location" make.tag *}
      if$
      
      s empty$
        'skip$
        { address empty$
            'skip$
            { ": " * }
          if$
          s *
        }
      if$
    }
  if$ 
}

FUNCTION {format.publisher.address}
{ publisher "publisher" make.tag 
  format.org.or.pub 
}

FUNCTION {format.organization.address}
{ organization "publisher" make.tag 
  format.org.or.pub 
}

FUNCTION {article}
{ output.bibitem
  
  author empty$
  institution empty$ 
  editor empty$
  and and
  journal empty$ 
  year empty$ 
  or or 
    {#0 'bother :=} 
    { 
      volume empty$ pages  empty$ %doi empty$ not
        or %and
        {#0 'bother :=}
        {#1 'bother :=}
      if$       
    } 
  if$   
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.atitle "title" output.check
  new.block
  crossref missing$
    {
      journal "jtitle" make.tag "journal" bibinfo.check
      emphasize
      "journal" output.check
      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.note output
  end.element
}
FUNCTION {book}
{ output.bibitem
  author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  title empty$ 
  year empty$ 
  or or 
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
      add.blank
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  crossref missing$
    { format.bvolume output
      new.block
      format.number.series output
      new.sentence
      format.publisher.address output
    }
    {
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  new.block
  format.note output
  end.element
}
FUNCTION {booklet}
{ output.bibitem
  author empty$
  title empty$ 
  year empty$ 
  howpublished empty$  
    or or or
    {#0 'bother :=} 
    {#1 'bother :=} 
  if$ 

  begin.element
  format.authors output
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  new.block
  howpublished "publisher" make.tag "howpublished" bibinfo.check output
  address "location" make.tag "address" bibinfo.check output
  new.block
  format.note output
  end.element
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  
  series empty$
  chapter empty$
  and
  year empty$ 
  or or
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  crossref missing$
    {
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.number.series output
      new.sentence
      format.publisher.address output
    }
    {
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.book.crossref output.nonnull
    }
  if$
  format.edition output
  new.block
  format.note output
  end.element
}

FUNCTION {incollection}
{ output.bibitem
  author empty$
  institution empty$ 
  and
  booktitle empty$ 
  title empty$
  year empty$ 
  or or or
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.ctitle "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      format.chapter.pages output
      new.sentence
      format.publisher.address output
      format.edition output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  new.block
  format.note output
  end.element
}
FUNCTION {inproceedings}
{ output.bibitem
  author empty$
  institution empty$ 
  and
  booktitle empty$ 
  series empty$
  and
  year empty$ 
  or or
    {#0 'bother :=} 
    {#1 'bother :=}
  if$ 
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.ctitle "title" output.check
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume output
      format.number.series output
      pages empty$
        'new.sentence
        'skip$
      if$
      format.pages output
      new.sentence
      publisher empty$
        { format.organization.address output }
        { organization "comment" make.tag "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.note output
  end.element
}
FUNCTION {conference} { inproceedings }
FUNCTION {manual}
{ output.bibitem
  author empty$
  organization empty$ 
  title empty$ 
  year empty$ 
  address  empty$ 
    or or or or
    {#0 'bother :=} 
    {#1 'bother :=} 
  if$ 
  begin.element
  format.authors output
  author format.key output
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  organization address new.block.checkb
  organization "publisher" make.tag "organization" bibinfo.check output
  address  "location" make.tag "address" bibinfo.check output
  format.edition output
  new.block
  format.note output
  end.element
}

FUNCTION {mastersthesis}
{ output.bibitem
  #0 'bother :=
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  new.block
  bbl.mthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  new.block
  format.note output
  end.element
}

FUNCTION {misc}
{ output.bibitem
  #0 'bother :=
  begin.element
  format.authors output
  author format.key output
  format.date "year" output.check
  date.block
  format.title output
  new.block
  howpublished "howpublished" bibinfo.check output
  new.block
  format.note output
  end.element
}
FUNCTION {phdthesis}
{ output.bibitem
  #0 'bother :=
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  new.block
  bbl.phdthesis format.thesis.type output.nonnull
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  new.block
  format.note output
  end.element
}

FUNCTION {proceedings}
{ output.bibitem
  author empty$
  institution empty$ 
  organization empty$ 
  editor empty$
  and and and
  booktitle empty$ 
  series empty$
  and 
    year empty$ 
  or or 
    {#0 'bother :=} 
    {
      publisher empty$ address empty$ not and
        {#0 'bother :=} 
        {#1 'bother :=}
      if$
    } 
  if$ 
  begin.element
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  editor format.key output
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  format.bvolume output
  format.number.series output
  new.sentence
  publisher empty$
    { format.organization.address output }
    { organization "comment" make.tag "organization" bibinfo.check output
      format.publisher.address output
    }
  if$
  new.block
  format.note output
  end.element
}

FUNCTION {techreport}
{ output.bibitem
  #0 'bother :=
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  new.block
  format.tr.number output.nonnull
  institution "institution" bibinfo.warn output
  address "address" bibinfo.check output
  new.block
  format.note output
  end.element
}

FUNCTION {unpublished}
{ output.bibitem
  #0 'bother :=
  begin.element
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  new.block
  format.note "note" output.check
  end.element
}

FUNCTION {default.type} { misc }
READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {format.lab.names}
{ 's :=
  "" 't :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$
      " " * bbl.etal *
    }
    { #2 <
        'skip$
        { s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            {
              " " * bbl.etal *
            }
            { bbl.and space.word * s #2 "{vv~}{ll}" format.name$
              * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        'key
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            'key
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.label}
{ editor empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        'key
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.short.authors}
{ author.editor.key.label
  'short.list :=
}

FUNCTION {calc.label}
{ calc.short.authors
  short.list
  "("
  *
  year duplicate$ empty$
  short.list key field.or.null = or
     { pop$ "" }
     'skip$
  if$
  *
  'label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{vv{ } }{ll{ }}{  f{ }}{  jj{ }}"
      format.name$ 't :=
      nameptr #1 >
        {
          "   "  *
          namesleft #1 = t "others" = and
            { "zzzzz" * }
            { t sortify * }
          if$
        }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}
FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {editor.sort}
{ editor empty$
    { key empty$
        { "to sort, need editor or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { editor sort.format.names }
  if$
}
FUNCTION {presort}
{ calc.label
  label sortify
  "    "
  *
  author.editor.sort
  #1 entry.max$ substring$
  'sort.label :=
  sort.label
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}
SORT
STRINGS { next.extra }
INTEGERS { last.extra.num number.label }
FUNCTION {initialize.extra.label.stuff}
{ #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'last.extra.num :=
  #0 'number.label :=
}
FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}
FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  extra.label 'next.extra :=
  extra.label
  duplicate$ empty$
    'skip$
    { "{\natexlab{" swap$ * "}}" * }
  if$
  'extra.label :=
  label extra.label * 'label :=
}
EXECUTE {initialize.extra.label.stuff}
ITERATE {forward.pass}
REVERSE {reverse.pass}
FUNCTION {bib.sort.order}
{ sort.label
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}
ITERATE {bib.sort.order}
SORT
FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" number.label int.to.str$ * "}" * write$ newline$
  "% BibTex style file: ba2.bst, 2026-08-03" write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
