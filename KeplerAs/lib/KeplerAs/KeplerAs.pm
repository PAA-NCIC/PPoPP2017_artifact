package KeplerAs::KeplerAs;

require 5.10.0;

use strict;
use Data::Dumper;
use KeplerAs::KeplerAsGrammar;
use File::Spec;
use Carp;

our $VERSION = '1.06';

my %relOffset  = map { $_ => 1 } qw(BRA SSY CAL PBK PCNT);

my %absOffset  = map { $_ => 1 } qw(JCAL);

my %jumpOp     = (%relOffset, %absOffset);

my %noDest     = map { $_ => 1 } qw(ST STG STS STL RED);

my %reuseSlots = (r8 => 1, r20 => 2, r39 => 4);

sub Assemble
{
    my ($file, $include, $doReuse, $nowarn) = @_;

    my $regMap = {};
    $file = Preprocess($file, $include, 0, $regMap);
    my $vectors = delete $regMap->{__vectors};
    my $regBank = delete $regMap->{__regbank};

    my $regCnt = 0;
    my $barCnt = 0;

    my ($lineNum, @instructs, %labels, $ctrl, @branches, %reuse);

    push @instructs, $ctrl = {};

    foreach my $line (split "\n", $file)
    {
        $lineNum++;

        next unless preProcessLine($line);

        if (my $inst = processAsmLine($line, $lineNum))
        {

            push @branches, @instructs+0 if exists $jumpOp{$inst->{op}};

            push @{$ctrl->{ctrl}}, $inst->{ctrl};

            $inst->{ctrl} = $ctrl;

            push @instructs, $inst;
            push @instructs, $ctrl = {} if ((@instructs & 7) == 0);
        }
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            $labels{$1} = @instructs+0;
        }
        else
        {
            die "badly formed line at $lineNum: $line\n";
        }
    }
    push @{$ctrl->{ctrl}}, 0x00;
    push @instructs, { op => 'BRA', inst => 'BRA 0xfffff8;' };
    while (@instructs & 7)
    {
        push @instructs, $ctrl = {} if ((@instructs & 7) == 0);
        push @{$ctrl->{ctrl}}, 0x00;
        push @instructs, { op => 'NOP', inst => 'NOP;' };
    }

    foreach my $i (@branches)
    {
        if ($instructs[$i]{inst} !~ m'(\w+);$' || !exists $labels{$1})
            { die "instruction has invalid label: $instructs[$i]{inst}"; }

        $instructs[$i]{jump} = $labels{$1};

        if (exists $relOffset{$instructs[$i]{op}})
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', (($labels{$1} - $i - 1) * 8) & 0xffffff/e; }
        else
            { $instructs[$i]{inst} =~ s/(\w+);$/sprintf '0x%06x;', ($labels{$1} * 8) & 0xffffff/e; }
    }

    foreach my $i (0 .. $#instructs)
    {
        next unless $i & 7;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            my $capData = parseInstruct($inst, $gram) or next;

            if ($doReuse)
            {
                my @r0 = getVecRegisters($vectors, $capData);


                if (@r0 && !exists $noDest{$op})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        if (my $reuse = $reuse{$slot})
                        {
                            delete $reuse->{$_} foreach @r0;
                        }
                    }
                }
                %reuse = () if exists $jumpOp{$op};

                if ($gram->{type}{reuse})
                {
                    foreach my $slot (keys %reuseSlots)
                    {
                        next unless exists $capData->{$slot};

                        my $r = $capData->{$slot};
                        next if $r eq 'RZ';
                        next if $r eq $capData->{r0}; # dont reuse if we're writing this reg in the same instruction

                        my $reuse = $reuse{$slot} ||= {};

                        if (my $p = $reuse->{$r})
                        {
                            $instructs[$p]{ctrl}{reuse}[($p & 7) - 1] |= $reuseSlots{$slot};

                        }
                        elsif (keys %$reuse > 2)
                        {
                            my $oldest = (sort {$reuse->{$a} <=> $reuse->{$b}} keys %$reuse)[0];
                            delete $reuse->{$oldest};
                        }
                        $reuse->{$r} = $i;
                    }
                }
            }
            elsif ($gram->{type}{reuse})
            {
                $ctrl->{reuse}[($i & 7) - 1] = genReuseCode($capData);
            }
            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    foreach my $r (sort keys %$regBank)
    {
        my $bank  = $regBank->{$r};
        my $avail = $regMap->{$r};
        foreach my $pos (0 .. $#$avail)
        {
            if ($bank == ($avail->[$pos] & 7))
            {
                $regMap->{$r} = 'R' . splice @$avail, $pos, 1;
                last;
            }
        }
    }

    my (%liveTime, %pairedBanks, %reuseHistory);
    foreach my $i (0 .. $#instructs)
    {
        next unless $i & 7;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            my $capData   = parseInstruct($inst, $gram) or next;
            my $reuseType = $gram->{type}{reuse};

            my (%addReuse, %delReuse);
            foreach my $slot (qw(r8 r20 r39))
            {
                my $r = $capData->{$slot} or next;
                next if $r eq 'RZ';

                my $liveR = ref $regMap->{$r} ? $r : $regMap->{$r};

                if (my $liveTime = $liveTime{$liveR})
                {
                    push @{$liveTime->[$#$liveTime]}, "$i $inst";
                }
                else
                {
                    warn "register used without initialization ($r): $inst\n" unless $nowarn;
                    push @{$liveTime{$liveR}}, [$i,$i];
                }

                my $slotHist  = $reuseHistory{$slot} ||= {};
                my $selfReuse = $reuseType ? exists $slotHist->{$r} : 0;


                if (!$selfReuse && ref $regMap->{$r})
                {
                    foreach my $slot2 (grep {$_ ne $slot && exists $capData->{$_}} qw(r8 r20 r39))
                    {
                        my $r2 = $capData->{$slot2};
                        next if $r2 eq 'RZ' || $r2 eq $r;

                        my $slotHist2 = $reuseHistory{$slot2} ||= {};


                        if (!$reuseType || !exists $slotHist2->{$r2})
                        {
                            if (ref $regMap->{$r2})
                            {
                                push @{$pairedBanks{$r}{pairs}}, $r2;
                                $pairedBanks{$r}{banks} ||= [];
                            }
                            else
                            {
                                my $bank = substr($regMap->{$r2},1) & 7;

                                $pairedBanks{$r}{bnkCnt}++ unless $pairedBanks{$r}{banks}[$bank]++;
                                $pairedBanks{$r}{pairs} ||= [];
                            }
                            $pairedBanks{$r}{useCnt}++;
                        }
                    }
                }
                if ($reuseType)
                {
                    if ($ctrl->{reuse}[($i & 7) - 1] & $reuseSlots{$slot})
                        { $addReuse{$slot} = $r; }
                    else
                        { $delReuse{$slot} = $r; }
                }
            }
            $reuseHistory{$_}{$addReuse{$_}} = 1    foreach keys %addReuse;
            delete $reuseHistory{$_}{$delReuse{$_}} foreach keys %delReuse;

            foreach my $r0 (getVecRegisters($vectors, $capData))
            {
                my $liveR = ref $regMap->{$r0} ? $r0 : $regMap->{$r0};

                if (exists $noDest{$op})
                {
                    if (my $liveTime = $liveTime{$liveR})
                    {
                        push @{$liveTime->[$#$liveTime]}, "$i $inst";
                    }
                    else
                    {
                        warn "register used without initialization ($r0): $inst\n" unless $nowarn;
                        push @{$liveTime{$liveR}}, [$i,$i];
                    }
                }
                elsif (my $liveTime = $liveTime{$liveR})
                {
                    if ($i > $liveTime->[$#$liveTime][1])
                    {
                        push @{$liveTime{$liveR}}, [$i,$i, "$i $inst"];
                    }
                }
                else
                {
                    push @{$liveTime{$liveR}}, [$i,$i, "$i $inst"];
                }
            }

            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    foreach my $r (sort {
                    $pairedBanks{$b}{bnkCnt} <=> $pairedBanks{$a}{bnkCnt} ||
                    $pairedBanks{$b}{useCnt} <=> $pairedBanks{$a}{useCnt} ||
                    $a cmp $b
                  } keys %pairedBanks)
    {
        my $banks = $pairedBanks{$r}{banks};
        my $avail = $regMap->{$r};


        BANK: foreach my $bank (sort {$banks->[$a] <=> $banks->[$b] || $a <=> $b } (0..7))
        {
            foreach my $pos (0 .. $#$avail)
            {
                if ($bank == ($avail->[$pos] & 7))
                {
                    $regMap->{$r} = 'R' . splice @$avail, $pos, 1;

                    $pairedBanks{$_}{banks}[$bank]++ foreach @{$pairedBanks{$r}{pairs}};
                    last BANK;
                }
            }
        }
    }
    foreach my $r (sort keys %$regMap)
    {
        if (ref($regMap->{$r}) eq 'ARRAY')
        {
            $regMap->{$r} = 'R' . shift @{$regMap->{$r}};
        }
    }

    foreach my $i (0 .. $#instructs)
    {
        next unless $i & 7;

        $instructs[$i]{orig} = $instructs[$i]{inst};
        $instructs[$i]{inst} =~ s/(?<!\.)\b(\w+)\b(?!\[)/ exists($regMap->{$1}) ? $regMap->{$1} : $1 /ge;

        my ($op, $inst, $ctrl) = @{$instructs[$i]}{qw(op inst ctrl)};

        my $match = 0;
        foreach my $gram (@{$grammar{$op}})
        {
            my $capData = parseInstruct($inst, $gram) or next;

            foreach my $r (qw(r0 r8 r20 r39))
            {
                next unless exists($capData->{$r}) && $capData->{$r} ne 'RZ';

                my $val = substr $capData->{$r}, 1;

                my @r0 = getVecRegisters($vectors, $capData);
                my @r8 = getAddrVecRegisters($vectors, $capData);

                my $regInc = $r eq 'r0' ? scalar(@r0) || 1 : 1;
                my $regInc = $r eq 'r8' ? scalar(@r8) || 1 : 1;

                if ($val + $regInc > $regCnt)
                {
                    $regCnt = $val + $regInc;
                }
            }
            if ($op eq 'BAR')
            {
                if (exists $capData->{i8w4})
                {
                    $barCnt = $capData->{i8w4}+1 if $capData->{i8w4}+1 > $barCnt;
                }
                elsif (exists $capData->{r8})
                {
                    $barCnt = 16;
                }
            }
            my ($code, $reuse) = genCode($op, $gram, $capData);
            $instructs[$i]{code} = $code;

            if ($gram->{type}{reuse})
                { $instructs[$i]{caps} = $capData; }
            else
                { $ctrl->{reuse}[($i & 7) - 1] = $reuse; }


            $match = 1;
            last;
        }
        unless ($match)
        {
            print "$_->{rule}\n\n" foreach @{$grammar{$op}};
            die "Unable to encode instruction: $inst\n";
        }
    }

    my (@codes, %reuseHistory, @exitOffsets, @ctaidOffsets, $ctaidzUsed);
    foreach my $i (0 .. $#instructs)
    {
        if ($i & 7)
        {
            push @codes, $instructs[$i]{code};
            my $code_dec= $instructs[$i]{code};
            my $code_hex = sprintf("0x%x", $code_dec);

            if ($instructs[$i]{caps})
            {
                registerHealth(\%reuseHistory, $instructs[$i]{ctrl}{reuse}[($i & 7) - 1], $instructs[$i]{caps}, $i * 8, "$instructs[$i]{inst} ($instructs[$i]{orig})", $nowarn);
            }
            if ($instructs[$i]{inst} =~ m'EXIT')
            {
                push @exitOffsets, (scalar(@codes)-1)*8;
            }
            elsif ($instructs[$i]{inst} =~ m'SR_CTAID\.(X|Y|Z)')
            {
                push @ctaidOffsets, (scalar(@codes)-1)*8;
                $ctaidzUsed = 1 if $1 eq 'Z';
            }
        }
        else
        {
            my ($ctrl, $ruse) = @{$instructs[$i]}{qw(ctrl reuse)};
            push @codes,
                ($ctrl->[0] <<  2) | ($ctrl->[1] << 10) | ($ctrl->[2] << 18) | # ctrl codes
                ($ctrl->[3] << 26) | ($ctrl->[4] << 34) | ($ctrl->[5] << 42) |
                ($ctrl->[6] << 50) | (0x0800000000000000);  # reuse codes
        }
    }

    return {
        RegCnt       => $regCnt,
        BarCnt       => $barCnt,
        ExitOffsets  => \@exitOffsets,
        CTAIDOffsets => \@ctaidOffsets,
        CTAIDZUsed   => $ctaidzUsed,
        ConflictCnt  => $reuseHistory{conflicts},
        ReuseCnt     => $reuseHistory{reuse},
        ReuseTot     => $reuseHistory{total},
        ReusePct     => ($reuseHistory{total} ? 100 * $reuseHistory{reuse} / $reuseHistory{total} : 0),
        KernelData   => \@codes,
    };
}

sub Test
{
    my ($fh, $printConflicts, $all) = @_;

    my @instructs;
    my %reuseHistory;
    my ($pass, $fail) = (0,0);

    while (my $line = <$fh>)
    {
        my (@ctrl, @reuse);

        next unless processSassCtrlLine($line, \@ctrl, \@reuse);

        foreach my $fileReuse (@reuse)
        {
            $line = <$fh>;

            my $inst = processSassLine($line) or next;

            $inst->{reuse} = $fileReuse;
            my $fileCode = $inst->{code};

            if (exists $relOffset{$inst->{op}})
            {
                $inst->{inst} =~ s/(0x[0-9a-f]+)/sprintf '0x%06x', ((hex($1) - $inst->{num} - 8) & 0xffffff)/e;
            }

            my $match = 0;
            foreach my $gram (@{$grammar{$inst->{op}}})
            {
                my $capData = parseInstruct($inst->{inst}, $gram) or next;
                my @caps;

                my ($code, $reuse) = genCode($inst->{op}, $gram, $capData, \@caps);

                registerHealth(\%reuseHistory, $reuse, $capData, $inst->{num}, $printConflicts ? $inst->{inst} : '') if $gram->{type}{reuse};

                $inst->{caps}      = join ', ', sort @caps;
                $inst->{codeDiff}  = $fileCode  ^ $code;
                $inst->{reuseDiff} = $fileReuse ^ $reuse;

                if ($code == $fileCode && $reuse == $fileReuse)
                {
                    $inst->{grade} = 'PASS';
                    push @instructs, $inst if $all;
                    $pass++;
                }
                else
                {
                    $inst->{grade} = 'FAIL';
                    push @instructs, $inst;
                    $fail++;
                }
                $match = 1;
                last;
            }
            unless ($match)
            {
                $inst->{grade}     = 'FAIL';
                $inst->{codeDiff}  = $fileCode;
                $inst->{reuseDiff} = $fileReuse;
                push @instructs, $inst;
                $fail++;
            }
        }
    }
    my %maxLen;
    foreach (@instructs)
    {
        $maxLen{$_->{op}} = length($_->{ins}) if length($_->{ins}) > $maxLen{$_->{op}};
    }
    my ($lastOp, $template);
    foreach my $inst (sort {
        $a->{op}        cmp $b->{op}        ||
        $a->{codeDiff}  <=> $b->{codeDiff}  ||
        $a->{reuseDiff} <=> $b->{reuseDiff} ||
        $a->{ins}       cmp $b->{ins}
        } @instructs)
    {
        if ($lastOp ne $inst->{op})
        {
            $lastOp   = $inst->{op};
            $template = "%s 0x%016x %x 0x%016x %x %5s%-$maxLen{$lastOp}s   %s\n";
            printf "\n%s %-18s %s %-18s %s %-5s%-$maxLen{$lastOp}s   %s\n", qw(Grad OpCode R opCodeDiff r Pred Instruction Captures);
        }
        printf $template, @{$inst}{qw(grade code reuse codeDiff reuseDiff pred ins caps)};
    }
    my $reusePct = $reuseHistory{total} ? 100 * $reuseHistory{reuse} / $reuseHistory{total} : 0;

    printf "\nRegister Bank Conflicts: %d, Reuse: %.1f% (%d/%d)\nOp Code Coverage Totals: Pass: $pass Fail: $fail\n",
        $reuseHistory{conflicts}, $reusePct, $reuseHistory{reuse}, $reuseHistory{total};

    return $fail;
}

sub Extract
{
    my ($in, $out, $params) = @_;

    my %paramMap;
    my %constants =
    (
        blockDimX => 'c[0x0][0x28]',
        blockDimY => 'c[0x0][0x2c]',
        blockDimZ => 'c[0x0][0x30]',
        gridDimX  => 'c[0x0][0x34]',
        gridDimY  => 'c[0x0][0x38]',
        gridDimZ  => 'c[0x0][0x3c]',
    );
    print $out "<CONSTANT_MAPPING>\n";

    foreach my $const (sort keys %constants)
    {
        print $out "    $const : $constants{$const}\n";
        $paramMap{$constants{$const}} = $const;
    }
    print $out "\n";

    foreach my $p (@$params)
    {
        my ($ord,$offset,$size,$align) = split ':', $p;

        if ($size > 4)
        {
            my $num = 0;
            $offset = hex $offset;
            while ($size > 0)
            {
                my $param = sprintf 'param_%d[%d]', $ord, $num;
                my $const = sprintf 'c[0x0][0x%x]', $offset;
                $paramMap{$const} = $param;
                print $out "    $param : $const\n";
                $size   -= 4;
                $offset += 4;
                $num    += 1;
            }
        }
        else
        {
            my $param = sprintf 'param_%d', $ord;
            my $const = sprintf 'c[0x0][%s]', $offset;
            $paramMap{$const} = $param;
            print $out "    $param : $const\n";
        }
    }
    print $out "</CONSTANT_MAPPING>\n\n";

    my %labels;
    my $labelnum = 1;

    my @data;
    FILE: while (my $line = <$in>)
    {
        my (@ctrl, @ruse);
        next unless processSassCtrlLine($line, \@ctrl, \@ruse);

        CTRL: foreach my $ctrl (@ctrl)
        {
            $line = <$in>;

            my $inst = processSassLine($line) or next CTRL;

            if (exists($jumpOp{$inst->{op}}) && $inst->{ins} =~ m'(0x[0-9a-f]+)')
            {
                my $target = hex($1);

                last FILE if $inst->{op} eq 'BRA' && ($target == $inst->{num}|| $target == $inst->{num}-8);

                my $label = $labels{$target};
                unless ($label)
                {
                    $label = $labels{$target} = "TARGET$labelnum";
                    $labelnum++;
                }
                $inst->{ins} =~ s/(0x[0-9a-f]+)/$label/;
            }
            $inst->{ins} =~ s/(c\[0x0\])\s*(\[0x[0-9a-f]+\])/ $paramMap{$1 . $2} || $1 . $2 /eg;

            $inst->{ctrl} = printCtrl($ctrl);

            push @data, $inst;
        }
    }
    foreach my $inst (@data)
    {
        print $out "$labels{$inst->{num}}:\n" if exists $labels{$inst->{num}};
        printf $out "%s %5s%s\n", @{$inst}{qw(ctrl pred ins)};
    }
}

my $CommentRe  = qr'^[\t ]*<COMMENT>.*?^\s*</COMMENT>\n?'ms;
my $IncludeRe  = qr'^[\t ]*<INCLUDE\s+file="([^"]+)"\s*/?>\n?'ms;
my $CodeRe     = qr'^[\t ]*<CODE(\d*)>(.*?)^\s*<\/CODE\1>\n?'ms;
my $ConstMapRe = qr'^[\t ]*<CONSTANT_MAPPING>(.*?)^\s*</CONSTANT_MAPPING>\n?'ms;
my $RegMapRe   = qr'^[\t ]*<REGISTER_MAPPING>(.*?)^\s*</REGISTER_MAPPING>\n?'ms;
my $ScheduleRe = qr'^[\t ]*<SCHEDULE_BLOCK>(.*?)^\s*</SCHEDULE_BLOCK>\n?'ms;
my $InlineRe   = qr'\[(\+|\-)(.+?)\1\]'ms;

sub IncludeFile
{
    my ($file, $include) = @_;
    my ($vol,$dir,$name) = File::Spec->splitpath($file);
    local $/;
    my $fh;
    if (!open $fh, $file)
    {
        open $fh, File::Spec->catpath(@$include, $name) or die "Could not open file for INCLUDE: $file ($!)\n";
    }
    my $content = <$fh>;
    close $fh;
    return $content;
}

sub Preprocess
{
    my ($file, $include, $debug, $regMap) = @_;

    my $constMap = {};
    my $removeRegMap;
    if ($regMap)
        { $removeRegMap = 1; }
    else
        { $regMap = {}; }

    1 while $file =~ s|$IncludeRe| IncludeFile($1, $include) |eg;

    $file =~ s|$CommentRe||g;

    1 while $file =~ s|$CodeRe|
        my $out = eval "package KeplerAs::KeplerAs::CODE; $2";
        $@ ? die("CODE:\n$2\n\nError: $@\n") : $out |eg;

    $file =~ s|$InlineRe|
        my ($type, $code) = ($1, $2);
        my $out = eval "package KeplerAs::KeplerAs::CODE; $code";
        $@ ? die("CODE:\n$code\n\nError: $@\n") : $type eq "+" ? $out : "" |eg;

    $file =~ s/$ConstMapRe/ setConstMap($constMap, $1) /eg;

    my @newFile;
    foreach my $line (split "\n", $file)
    {
        if ($line !~ m'^\s*(?:#|//).*')
        {
            $line =~ s|(\w+(?:\[\d+\])?)| exists $constMap->{$1} ? $constMap->{$1} : $1 |eg;
        }
        push @newFile, $line;
    }
    $file = join "\n", @newFile;

    $file =~ s/$RegMapRe/ setRegisterMap($regMap, $1); $removeRegMap ? '' : $& /eg;

    my @schedBlocks = $file =~ /$ScheduleRe/g;

    foreach my $i (0 .. $#schedBlocks)
    {
        $schedBlocks[$i] = replaceXMADs($schedBlocks[$i]);

        $schedBlocks[$i] = Scheduler($schedBlocks[$i], $i+1, $regMap, $debug);
    }

    $file =~ s|$ScheduleRe| shift @schedBlocks |eg;

    return $file;
}

my %srcReg   = map { $_ => 1 } qw(r8 r20 r39 p12 p29 p39 X);
my %destReg  = map { $_ => 1 } qw(r0 p0 p3 p45 p48 CC);
my %regops   = (%srcReg, %destReg);
my @itypes   = qw(class lat rlat tput dual);

sub Scheduler
{
    my ($block, $blockNum, $regMap, $debug) = @_;

    my $vectors = $regMap->{__vectors};
    my $lineNum = 0;

    my (@instructs, @comments, $ordered, $first);
    foreach my $line (split "\n", $block)
    {
        $lineNum++;

        unless (preProcessLine($line))
        {
            push @comments, $line if $line =~ m'\S';
            next;
        }

        if (my $inst = processAsmLine($line, $lineNum))
        {
            $inst->{first}   = !$first++ && ($inst->{ctrl} & 0x1f800) ? 0 : 1;

            $inst->{exeTime} = 0;
            $inst->{order}   = $ordered++ if $ordered;
            push @instructs, $inst;
        }
        elsif ($line =~ m'^([a-zA-Z]\w*):')
        {
            die "SCHEDULE_BLOCK's cannot contain labels. block: $blockNum line: $lineNum\n";
        }
        elsif ($line =~ m'^<ORDERED>')
        {
            die "you cannot use nested <ORDERED> tags" if $ordered;
            $ordered = 1;
        }
        elsif ($line =~ m'^</ORDERED>')
        {
            die "missing opening <ORDERED> for closing </ORDERED> tag" if !$ordered;
            $ordered = 0;
        }
        else
        {
            die "badly formed line at block: $blockNum line: $lineNum: $line\n";
        }
    }

    my (%writes, %reads, @ready, @schedule, $orderedParent);
    foreach my $instruct (@instructs)
    {
        my $match = 0;
        foreach my $gram (@{$grammar{$instruct->{op}}})
        {
            my $capData = parseInstruct($instruct->{inst}, $gram) or next;
            my (@dest, @src);

            @{$instruct}{@itypes} = @{$gram->{type}}{@itypes};

            push @src, $instruct->{predReg} if $instruct->{pred};

            if ($instruct->{op} =~ m'P2R|R2P' && $capData->{i20w7})
            {
                my $list = $instruct->{op} eq 'R2P' ? \@dest : \@src;
                my $mask = hex($capData->{i20w7});
                foreach my $p (0..6)
                {
                    if ($mask & (1 << $p))
                    {
                        push @$list, "P$p";
                    }
                    elsif ($instruct->{op} eq 'R2P')
                    {
                        push @src, "P$p";
                    }
                }
            }

            foreach my $operand (grep { exists $regops{$_} } sort keys %$capData)
            {
                my $list = exists($destReg{$operand}) && !exists($noDest{$instruct->{op}}) ? \@dest : \@src;

                my $badVal = substr($operand,0,1) eq 'r' ? 'RZ' : 'PT';

                if ($capData->{$operand} ne $badVal)
                {
                    push @$list,
                        $operand eq 'r0' ? map(getRegNum($regMap, $_), getVecRegisters($vectors, $capData)) :
                        $operand eq 'r8' ? map(getRegNum($regMap, $_), getAddrVecRegisters($vectors, $capData)) :
                        $operand eq 'CC' ? 'CC' :
                        $operand eq 'X'  ? 'CC' :
                        getRegNum($regMap, $capData->{$operand});
                }
            }
            $instruct->{const} = 1 if exists($capData->{c20}) || exists($capData->{c39});

            foreach my $src (grep { exists $writes{$_} } @src)
            {
                my $regLatency = $src eq $instruct->{predReg} ? 0 : $instruct->{rlat};

                foreach my $parent (@{$writes{$src}})
                {
                    my $latency = $src =~ m'^P\d' ? 13 : $parent->{lat};
                    push @{$parent->{children}}, [$instruct, $latency - $regLatency];
                    $instruct->{parents}++;

                    last unless $parent->{pred};
                }
            }

            foreach my $dest (grep { exists $reads{$_} } @dest)
            {
                foreach my $reader (@{$reads{$dest}})
                {
                    push @{$reader->{children}}, [$instruct, 0];
                    $instruct->{parents}++;
                }
                delete $reads{$dest} unless $instruct->{pred};
            }

            if ($instruct->{order})
            {
                if ($orderedParent)
                {
                    push @{$orderedParent->{children}}, [$instruct, 0];
                    $instruct->{parents}++;
                }
                $orderedParent = $instruct;
            }
            elsif ($orderedParent)
                {  $orderedParent = 0; }

            unshift @{$writes{$_}}, $instruct foreach @dest;

            push @{$reads{$_}}, $instruct foreach @src;

            push @ready, $instruct if !exists $instruct->{parents};

            $match = 1;
            last;
        }
        die "Unable to recognize instruction at block: $blockNum line: $lineNum: $instruct->{inst}\n" unless $match;
    }
    %writes = ();
    %reads  = ();

    if (@ready)
    {
        my $readyParent = { children => [ map { [ $_, 1 ] } @ready ], inst => "root" };

        countUniqueDescendants($readyParent, {});
        updateDepCounts($readyParent, {});

        @ready = sort {
            $a->{first}   <=> $b->{first}  ||
            $b->{deps}    <=> $a->{deps}   ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;

        if ($debug)
        {
            print  "0: Initial Ready List State:\n\tf,ext,stl,mix,dep,lin, inst\n";
            printf "\t%d,%3s,%3s,%3s,%3s,%3s, %s\n", @{$_}{qw(first exeTime stall mix deps lineNum inst)} foreach @ready;
        }
    }

    my $clock = 0;
    while (my $instruct = shift @ready)
    {
        my $stall = $instruct->{stall};

        if (@schedule && $stall < 16)
        {
            my $prev = $schedule[$#schedule];

            $prev->{ctrl} &= $stall > 4 ? 0x1ffe0 : 0x1fff0;
            $prev->{ctrl} |= $stall;
            $clock += $stall;
        }
        else
        {
            $instruct->{ctrl} &= 0x1fff0;
            $instruct->{ctrl} |= 1;
            $clock += 1;
        }
        print "$clock: $instruct->{inst}\n" if $debug;

        push @schedule, $instruct;

        if (my $children = $instruct->{children})
        {
            foreach (@$children)
            {
                my ($child, $latency) = @$_;

                my $earliest = $clock + $latency;
                $child->{exeTime} = $earliest if $child->{exeTime} < $earliest;

                print "\t\t$child->{exeTime},$child->{parents} $child->{inst}\n" if $debug;

                push @ready, $child if --$child->{parents} < 1;
            }
            delete $instruct->{children};
        }

        foreach my $ready (@ready)
        {
            $stall = $ready->{exeTime} - $clock;
            $stall = 1 if $stall < 1;

            if ($ready->{class} eq $instruct->{class})
            {
                $stall = $ready->{tput} if $stall < $ready->{tput};
            }
            elsif ($ready->{dual} && !$instruct->{dual} && $instruct->{tput} <= 2 &&
                   $stall == 1 && $ready->{exeTime} <= $clock && !($ready->{const} && $instruct->{const}))
            {
                $stall = 0;
            }
            $ready->{stall} = $stall;

            $ready->{mix} = $ready->{class} ne $instruct->{class} || 0;
        }

        @ready = sort {
            $a->{first}   <=> $b->{first}  ||
            $a->{stall}   <=> $b->{stall}  ||
            $b->{mix}     <=> $a->{mix}    ||
            $b->{deps}    <=> $a->{deps}   ||
            $a->{lineNum} <=> $b->{lineNum}
            } @ready;

        if ($debug)
        {
            print  "\tf,ext,stl,mix,dep,lin, inst\n";
            printf "\t%d,%3s,%3s,%3s,%3s,%3s, %s\n", @{$_}{qw(f exeTime stall mix deps lineNum inst)} foreach @ready;
        }
    }

    my $out;
    $out .= join('', printCtrl($_->{ctrl}), @{$_}{qw(space inst comment)}, "\n") foreach @schedule;
    return $out;
}

sub setConstMap
{
    my ($constMap, $constMapText) = @_;

    foreach my $line (split "\n", $constMapText)
    {
        $line =~ s|^\s+||;
        $line =~ s{(?:#|//).*}{};
        $line =~ s|\s+$||;
        next unless $line =~ m'\S';

        my ($name, $value) = split '\s*:\s*', $line;

        $constMap->{$name} = $value;
    }
    return;
}

sub setRegisterMap
{
    my ($regMap, $regmapText) = @_;

    my $vectors = $regMap->{__vectors} ||= {};
    my $regBank = $regMap->{__regbank} ||= {};
    my %aliases;

    foreach my $line (split "\n", $regmapText)
    {
        $line =~ s|^\s+||;
        $line =~ s{(?:#|//).*}{};
        $line =~ s|\s+$||;
        next unless $line =~ m'\S';

        my $auto  = $line =~ /~/;
        my $share = $line =~ /=/;

        my ($regNums, $regNames) = split '\s*[:~=]\s*', $line;

        my (@numList, @nameList, %vecAliases);
        foreach my $num (split '\s*,\s*', $regNums)
        {
            my ($start, $stop) = split '\s*\-\s*', $num;
            die "REGISTER_MAPPING Error: Bad register number or range: $num\nLine: $line\nFull Context:\n$regmapText\n" if grep m'\D', $start, $stop;
            push @numList, ($start .. $stop||$start);
        }
        foreach my $fullName (split '\s*,\s*', $regNames)
        {
            if ($fullName =~ m'^(\w+)<((?:\d+(?:\s*\-\s*\d+)?\s*\|?\s*)+)>(\w*)(?:\[([0-3])\])?$')
            {
                my ($name1, $name2, $bank) = ($1, $3, $4);
                foreach (split '\s*\|\s*', $2)
                {
                    my ($start, $stop) = split '\s*\-\s*';
                    foreach my $r (map "$name1$_$name2", $start .. $stop||$start)
                    {
                        $aliases{$r} = "$name1$name2" unless exists $aliases{$r};
                        push @nameList, $r;
                        $regBank->{$r} = $bank if $auto && defined $bank;
                        warn "Cannot request a bank for a fixed register range: $fullName\n" if !$auto && defined $bank;
                    }
                }
            }
            elsif ($fullName =~ m'^(\w+)(?:\[([0-3])\])?$')
            {
                push @nameList, $1;
                $regBank->{$1} = $2 if $auto && defined $2;
                warn "Cannot request a bank for a fixed register range: $fullName\n" if !$auto && defined $2;
            }
            else
            {
                die "Bad register name: '$fullName' at: $line\n";
            }
        }
        die "Missmatched register mapping at: $line\n" if !$share && @numList < @nameList;
        die "Missmatched register mapping at: $line\n" if $share && @numList > 1;

        my $i = 0;
        while ($i < $#numList-1)
        {
            last if $numList[$i] + 1 != $numList[$i+1];
            $i++;
        }
        my $ascending = $i+1 == $#numList;

        foreach my $n (0..$#nameList)
        {
            die "register defined twice: $nameList[$n]" if exists $regMap->{$nameList[$n]};

            if ($auto)
            {
                $regMap->{$nameList[$n]} = \@numList;
            }
            elsif ($share)
            {
                $regMap->{$nameList[$n]} = 'R' . $numList[0];
            }
            else
            {
                $regMap->{$nameList[$n]} = 'R' . $numList[$n];
                if ($ascending && ($numList[$n] & 1) == 0)
                {
                    my $end = $n + ($numList[$n] & 2 || $n + 3 > $#nameList ? 1 : 3);
                    if ($end <= $#nameList)
                    {
                        $vectors->{$nameList[$n]} = [ @nameList[$n .. $end] ];
                        if (exists $aliases{$nameList[$n]} && !exists $regMap->{$aliases{$nameList[$n]}})
                        {
                            $regMap->{$aliases{$nameList[$n]}}  = $regMap->{$nameList[$n]};
                            $vectors->{$aliases{$nameList[$n]}} = $vectors->{$nameList[$n]};
                            delete $aliases{$nameList[$n]};
                        }
                    }
                }
            }
        }
    }
}

sub preProcessLine
{
    $_[0] =~ s|^\s+||;

    my $val = shift;

    $val =~ s{(?:#|//).*}{};

    return $val =~ m'\S';
}

sub countUniqueDescendants
{
    my ($node, $edges) = @_;


    if (my $children = $node->{children})
    {
        foreach my $child (grep $_->[1], @$children) # skip WaR deps and traversed edges
        {
            next if $edges->{"$node->{lineNum}^$child->[0]{lineNum}"}++;

            $node->{deps}{$_}++ foreach countUniqueDescendants($child->[0], $edges);
        }
    }
    else
    {
        return $node->{lineNum};
    }
    return ($node->{lineNum}, keys %{$node->{deps}});
}
sub updateDepCounts
{
    my ($node, $edges) = @_;


    if (my $children = $node->{children})
    {
        foreach my $child (@$children)
        {
            next if $edges->{"$node->{lineNum}^$child->[0]{lineNum}"}++;
            updateDepCounts($child->[0], $edges);
        }
    }
    $node->{deps} = ref $node->{deps} ? keys %{$node->{deps}} : $node->{deps}+0;
}

sub registerHealth
{
    my ($reuseHistory, $reuseFlags, $capData, $instAddr, $inst, $nowarn) = @_;

    my (@banks, @conflicts);

    foreach my $slot (qw(r8 r20 r39))
    {
        my $r = $capData->{$slot} or next;
        next if $r eq 'RZ';

        my $slotHist = $reuseHistory->{$slot} ||= {};

        $reuseHistory->{total}++;

        if (exists $slotHist->{$r})
        {
            $reuseHistory->{reuse}++;
        }
        else
        {
            my $bank = substr($r,1) & 7;

            if ($banks[$bank] && $banks[$bank] ne $r)
            {
                push @conflicts, $banks[$bank] if !@conflicts;
                push @conflicts, $r;

                $reuseHistory->{conflicts}++;
            }
            $banks[$bank] = $r;
        }

        if ($reuseFlags & $reuseSlots{$slot})
            { $slotHist->{$r} = 1; }
        else
            { delete $slotHist->{$r};  }
    }
    if ($inst && @conflicts && !$nowarn)
    {
        printf "CONFLICT at 0x%04x (%s): $inst\n", $instAddr, join(',', @conflicts);
    }
    return scalar @conflicts;
}

1;

__END__

=head1 NAME

KeplerAs::KeplerAs - Assembler for NVIDIA Maxwell architecture

=head1 SYNOPSIS

    KeplerAs.pl [opts]

=head1 DESCRIPTION

See the documentation at: https://github.com/NervanaSystems/KeplerAs

=head1 SEE ALSO

See the documentation at: https://github.com/NervanaSystems/KeplerAs


=head1 AUTHOR

Scott Gray, E<lt>sgray@nervanasys.com<gt>

=head1 COPYRIGHT AND LICENSE

The MIT License (MIT)

Copyright (c) 2014 Scott Gray

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

=cut
